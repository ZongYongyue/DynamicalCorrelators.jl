function add_single_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor))
    if length(codomain(tensor)) < length(domain(tensor))
        util = isomorphism(storagetype(tensor), ou * codomain(tensor), codomain(tensor))
        tensors = util * tensor
    elseif length(codomain(tensor)) > length(domain(tensor))
        util = isomorphism(storagetype(tensor), domain(tensor), domain(tensor) * ou)
        tensors = tensor * util
    else 
        throw(ArgumentError("invalid operator"))
    end
    return tensors
end

function cart2polar(point::AbstractArray)
    r = norm(point) 
    θ = atan(point[2], point[1])
    ϕ = length(point) == 3 ? acos(point[3]/r) : π/2
    return r, θ, ϕ
end

function phase_by_polar(theta::AbstractVector, phi::AbstractVector, phases::AbstractVector)
    function _phase_by_polar(bond::Bond)
        _, θ, ϕ = cart2polar(rcoordinate(bond))
        for i in eachindex(phases)
            any(≈(θ), theta[i]) && any(≈(ϕ), phi[i]) && return phases[i]
        end
    end
    return _phase_by_polar
end

function sort_by_distance(latt::CustomLattice, ij)
    is = ij[length(ij)÷2:-1:1]
    js = ij[length(ij)÷2+1:1:length(ij)]
    ks = is[2:end]
    ls = js[1:end-1]
    r = []
    for i in eachindex(is)
        push!(r, [is[i], js[i], norm(latt.lattice[is[i]]-latt.lattice[js[i]])])
    end
    for i in eachindex(ks)
        push!(r, [ks[i], ls[i], norm(latt.lattice[ks[i]]-latt.lattice[ls[i]])])
    end
    sort!(r, by=x->x[3])
    is = zeros(Int64, length(r))
    js = zeros(Int64, length(r))
    ds = zeros(Float64, length(r))
    for i in eachindex(is)
        is[i] = Int(r[i][1])
        js[i] = Int(r[i][2])
        ds[i] = r[i][3]
    end
    return is, js, ds
end

function transfer_left(v::AbstractTensorMap{T, S, 4, 1}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {T, S}
    check_unambiguous_braiding(space(v, 4))
    @plansor v[-1 -2 -3 -4; -5] := v[1 2 3 4; 5] * A[5 6; -5] * τ[4 7; 6 -4] * τ[3 8; 7 -3] * τ[2 9; 8 -2] * conj(Ab[1 9; -1])
end

function transfer_left(v::AbstractTensorMap{T, S, 3, 1}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {T, S}
    check_unambiguous_braiding(space(v, 2))
    @plansor v[-1 -2 -3; -4] := v[1 2 3; 4] * A[4 5; -4] * τ[3 6; 5 -3] * τ[2 7; 6 -2] * conj(Ab[1 7; -1])
end


function contract_MPO(mpo1::FiniteMPO{<:MPOTensor}, mpo2::FiniteMPO{<:MPOTensor})
    T = promote_type(scalartype(mpo1[1]), scalartype(mpo2[end]))
    F1 = fuser(T, right_virtualspace(mpo2[1]), right_virtualspace(mpo1[1]))
    F2 = fuser(T, left_virtualspace(mpo2[end]), left_virtualspace(mpo1[end]))
    ops = map(fuse_mul_mpo, parent(mpo1)[2:end-1], parent(mpo2)[2:end-1])
    @plansor O₁[-1; -2 -3] :=  mpo2[1][1 2; -2 3] * mpo1[1][1 -1; 2 4] * conj(F1[-3; 3 4])
    @plansor O₂[-3 -1; -2] := F2[-3; 3 4] * mpo2[end][3 2; -2 1] * mpo1[end][4 -1; 2 1] 
    O₁, O₂ = add_single_util_leg(O₁), add_single_util_leg(O₂)
    return changebonds!(FiniteMPO([O₁;ops;O₂]), SvdCut(; trscheme=notrunc()))
end