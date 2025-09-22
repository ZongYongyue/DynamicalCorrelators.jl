"""
    evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); 
                        filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
"""
function evolve_mps(H::MPOHamiltonian, ts::AbstractVector, rho_mps::FiniteMPS=convert(FiniteMPS, identityMPO(H)); 
                    filename::String="default_expiHt_ψ.jld2", 
                    save_id::AbstractArray=[length(ts),], 
                    verbose::Bool=true, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    start_time, record_start = now(), now()
    verbose && println("[1/$(length(ts))] t = $(ts[1]) ", " | Started:", Dates.format(start_time, "d.u yyyy HH:MM"))
    flush(stdout)
    envs = environments(rho_mps, H)
    jldopen(filename, "w") do f
        f["ts"] = ts
        if 1 in save_id
            f["t=$(ts[1])"] = rho_mps
        end
    end
    for i in 2:length(ts)
        alg = i > n ? tdvp1 : tdvp2
        rho_mps, envs = timestep(rho_mps, H, 0, ts[i]-ts[i-1], alg, envs)
        current_time = now()
        verbose && println("[$i/$(length(ts))] t = $(ts[i]) ", " | duration:", Dates.canonicalize(current_time-start_time))
        flush(stdout)
        jldopen(filename, "a") do f
            if i in save_id
                f["t=$(ts[i])"] = rho_mps
            end
        end
        start_time = current_time
    end
    verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    return rho_mps
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in zero temperature
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(H), length(indices), length(times))
    @sync @distributed for id in indices
        start_time, record_start = now(), now()
        idx = id <= length(H) ? id : (id - length(H))
        ket = chargedMPS(op, gs, idx)
        gf[:,idx,1] = id <= length(H) ? [-im*exp(im*gsenergy*times[1])*dot(chargedMPS(op, gs, i), ket) for i in 1:length(H)] : [-im*exp(-im*gsenergy*times[1])*dot(ket, chargedMPS(op, gs, i)) for i in 1:length(H)]
        filename = joinpath(gf_path, "gf_tmax=$(times[end])_id=$(id).jld2")
        if isfile(filename)
            gfb = load(filename)
            for k in 2:length(times)
                if "pro_$(k)" in collect(keys(gfb))
                    gf[:,id,k] = gfb["pro_$(k)"]
                else
                    @warn "Key 'pro_$(k)' not found in $(filename)"
                end
            end
            verbose && println("gf_tmax=$(times[end])_id=$(id).jld2 has existed!")
            flush(stdout)
            continue
        else
            jldopen(filename, "w") do f
                f["pro_1"] = gf[:,idx,1]
            end
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        flush(stdout)
        envs = environments(ket, H)
        for k in 2:length(times)
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[k]-times[k-1], alg, envs)
            for i in 1:length(H)
                gf[i,idx,k] = (id <= length(H)) ? -im*exp(im*gsenergy*times[k])*dot(chargedMPS(op, gs, i), ket) : -im*exp(-im*gsenergy*times[k])*dot(ket, chargedMPS(op, gs, i))
            end
            current_time = now()
            verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(k)"] = gf[:,idx,k]
            end
            start_time = current_time
        end
        @everywhere begin
            ket = nothing
            envs = nothing
            GC.gc()
        end
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, length(H), length(indices), length(times))
    gfs .= gf
    return gf
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in finite temperature
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::String="./beta=$(beta)",
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme),
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gf = SharedArray{ComplexF64, 3}(length(H), length(indices), length(times))
    Z = dot(rho, rho)
    !isdir(rho_path)&& mkdir(rho_path)
    rho_filename = joinpath(rho_path, "rho_β=$(beta)_tmax=$(times[end]).jld2")
    if isfile(rho_filename)
        rhos = load(rho_filename, "rhos")
        verbose && println("rhos is successfully loaded")
        flush(stdout)
    else
        rhos = Vector{FiniteSuperMPS}(undef, length(times))
        rhos[1] = rho
        env = environments(rho, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            rho, env = timestep(rho, H, 0, times[i]-times[i-1], alg, env)
            verbose && println("[$(i-1)/$(length(times)-1)] evolves t=$(times[i]) of rho ", Dates.format(now(), "d.u yyyy HH:MM"))
            rhos[i] = rho
            flush(stdout)
        end
        save(rho_filename, "rhos", rhos)
    end
    env = nothing
    @sync @distributed for id in indices
        start_time, record_start = now(), now()
        idx = id <= length(H) ? id : (id - length(H))
        ket = chargedMPS(op, rhos[1], idx)
        gf[:,idx,1] = (id <= length(H)) ? [dot(chargedMPS(op, rhos[1], i), ket)/Z for i in 1:length(H)] : [conj(dot(chargedMPS(op, rhos[1], i), ket))/Z for i in 1:length(H)]
        flush(stdout)
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2")
        if isfile(filename)
            gfb = load(filename)
            for k in 2:length(times)
                if "pro_$(k)" in collect(keys(gfb))
                    gf[:,id,k] = gfb["pro_$(k)"]
                else
                    @warn "Key 'pro_$(k)' not found in $(filename)"
                end
            end
            verbose && println("gf_β=$(beta)_tmax=$(times[end])_id=$(id).jld2 has existed!")
            flush(stdout)
            continue
        else
            jldopen(filename, "w") do f
                f["pro_1"] = gf[:,idx,1]
            end
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(id) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for k in 2:length(times)
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[k]-times[k-1], alg, envs)
            for i in 1:length(H)
                gf[i,idx,k] = (id <= length(H)) ? dot(chargedMPS(op, rhos[k], i), ket)/Z : dot(ket, chargedMPS(op, rhos[k], i))/Z
            end
            current_time = now()
            verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(id) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(k)"] = gf[:,idx,k]
            end
            start_time = current_time
        end
        @everywhere begin
            ket = nothing
            envs = nothing
            GC.gc()
        end
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = zeros(ComplexF64, length(H), length(indices), length(times))
    gfs .= -im*gf
    return gfs
end

"""
    dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, mps::AbstractVector{<:FiniteNormalMPS};
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in zero temperature
"""
function dcorrelator(gs::FiniteNormalMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true, 
                    gf_path::String="./", 
                    times::AbstractRange=0:0.05:5.0, 
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme),
                    isfermion::Bool = true
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    mps = [[chargedMPS(ops[1], gs, i) for i in 1:length(H)];[chargedMPS(ops[2], gs, i) for i in 1:length(H)]]
    gsenergy = expectation_value(gs, H)
    gf = SharedArray{ComplexF64, 3}(length(H), 2*length(H), length(times))
    @sync @distributed for j in 1:2*length(H)
        start_time, record_start = now(), now()
        ket = mps[j]
        gf[:,j,1] = j <= length(H) ? [-im*exp(im*gsenergy*times[1])*dot(bra, ket) for bra in mps[1:length(H)]] : [-im*exp(im*gsenergy*times[1])*dot(ket, bra) for bra in mps[(length(H)+1):end]]
        flush(stdout)
        filename = joinpath(gf_path, "gf_tmax=$(times[end])_id=$(j).jld2")
        if isfile(filename)
            gfb = load(filename)
            for k in 2:length(times)
                if "pro_$(k)" in collect(keys(gfb))
                    gf[:,j,k] = gfb["pro_$(k)"]
                else
                    @warn "Key 'pro_$(k)' not found in $(filename)"
                end
            end
            verbose && println("gf_tmax=$(times[end])_id=$(j).jld2 has loaded!")
            flush(stdout)
            continue
        else
            jldopen(filename, "w") do f
                f["pro_1"] = gf[:,j,1]
            end
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(j) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for k in 2:length(times)
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[k]-times[k-1], alg, envs)
            for i in 1:(length(mps)÷2)
                gf[i,j,k] = (j <= length(H)) ? -im*exp(im*gsenergy*times[k])*dot(mps[1:length(H)][i], ket) : -im*exp(-im*gsenergy*times[k])*dot(ket, mps[(length(H)+1):end][i])
            end
            current_time = now()
            verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(j) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(k)"] = gf[:,j,k]
            end
            start_time = current_time
        end
        @everywhere begin
            ket = nothing
            envs = nothing
            GC.gc()
        end
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = isfermion ? (gf[:,1:length(H),:] .+ gf[:,(length(H)+1):2*length(H),:]) : (gf[:,1:length(H),:] .- gf[:,(length(H)+1):2*length(H),:])
    return gfs
end

"""
    dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, op::AbstractTensorMap, indices::AbstractArray;
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme)
                    )
    Dynamical correlations in finite temperature
"""
function dcorrelator(rho::FiniteSuperMPS, H::MPOHamiltonian, ops::Tuple{<:AbstractTensorMap, <:AbstractTensorMap};
                    verbose=true, 
                    gf_path::String="./",   
                    times::AbstractRange=0:0.05:5.0, 
                    beta::Union{Number, Missing}=missing,
                    rho_path::String="./beta=$(beta)",
                    n::Integer=3, 
                    trscheme=truncerr(1e-3),
                    tdvp1 = DefaultTDVP,
                    tdvp2 = DefaultTDVP2(trscheme),
                    isfermion::Bool = true
                    )
    !isdir(gf_path)&& mkdir(gf_path)
    gf = SharedArray{ComplexF64, 3}(length(H), 2*length(H), length(times))
    Z = dot(rho, rho)
    !isdir(rho_path)&& mkdir(rho_path)
    rho_filename = joinpath(rho_path, "rho_β=$(beta)_tmax=$(times[end]).jld2")
    if isfile(rho_filename)
        rhos = load(rho_filename, "rhos")
        verbose && println("rhos is successfully loaded")
        flush(stdout)
    else
        rhos = Vector{FiniteSuperMPS}(undef, length(times))
        rhos[1] = rho
        env = environments(rho, H)
        for i in 2:length(times)
            alg = i > n ? tdvp1 : tdvp2
            rho, env = timestep(rho, H, 0, times[i]-times[i-1], alg, env)
            verbose && println("[$(i-1)/$(length(times)-1)] evolves t=$(times[i]) of rho ")
            rhos[i] = rho
            flush(stdout)
        end
        save(rho_filename, "rhos", rhos)
    end
    env = nothing
    @sync @distributed for j in 1:2*length(H)
        start_time, record_start = now(), now()
        ket = j <= length(H) ? chargedMPS(ops[1], rhos[1], j) : chargedMPS(ops[2], rhos[1], j-length(H))
        gf[:,j,1] = (j <= length(H)) ? [dot(chargedMPS(ops[1], rhos[1], i), ket)/Z for i in 1:length(H)] : [conj(dot(chargedMPS(ops[2], rhos[1], i), ket))/Z for i in 1:length(H)]
        flush(stdout)
        filename = joinpath(gf_path, "gf_β=$(beta)_tmax=$(times[end])_id=$(j).jld2")
        if isfile(filename)
            gfb = load(filename)
            for k in 2:length(times)
                if "pro_$(k)" in collect(keys(gfb))
                    gf[:,j,k] = gfb["pro_$(k)"]
                else
                    @warn "Key 'pro_$(k)' not found in $(filename)"
                end
            end
            verbose && println("gf_β=$(beta)_tmax=$(times[end])_id=$(j).jld2 has loaded!")
            flush(stdout)
            continue
        else
            jldopen(filename, "w") do f
                f["pro_1"] = gf[:,j,1]
            end
        end
        verbose && println("[1/$(length(times))] Started: time evolves 0 of ket$(j) ", Dates.format(start_time, "d.u yyyy HH:MM"))
        envs = environments(ket, H)
        for k in 2:length(times)
            alg = k > n ? tdvp1 : tdvp2
            ket, envs = timestep(ket, H, 0, times[k]-times[k-1], alg, envs)
            if j <= length(H)
                for i in 1:length(H)
                    gf[i,j,k] = dot(chargedMPS(ops[1], rhos[k], i), ket)/Z
                end
            else
                for i in 1:length(H)
                    gf[i,j,k] = dot(ket, chargedMPS(ops[2], rhos[k], i))/Z
                end
            end
            current_time = now()
            verbose && println("[$(k)/$(length(times))] time evolves $(times[k]) of ket$(j) ", " | duration:", Dates.canonicalize(current_time-start_time))
            flush(stdout)
            jldopen(filename, "a") do f
                f["pro_$(k)"] = gf[:,j,k]
            end
            start_time = current_time
        end
        @everywhere begin
            ket = nothing
            envs = nothing
            GC.gc()
        end
        verbose && println("Ended: ", Dates.format(now(), "d.u yyyy HH:MM"), " | total duration: ", Dates.canonicalize(now()-record_start))
    end
    gfs = isfermion ? (gf[:,1:length(H),:] .+ gf[:,(length(H)+1):2*length(H),:]) : (gf[:,1:length(H),:] .- gf[:,(length(H)+1):2*length(H),:])
    return -im*gfs
end
