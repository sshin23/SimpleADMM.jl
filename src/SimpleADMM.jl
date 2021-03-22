module SimpleADMM

import Printf: @sprintf
import LightGraphs: Graph, add_edge!, neighbors, nv
import SimpleNLModels: Expression, Model, variable, parameter, constraint, objective,
    num_variables, num_constraints, func, deriv, optimize!, instantiate!, index

export ADMMModel, iterate!, optimize!

mutable struct SubModel
    model::Model

    x_V_orig
    x_V_sub
    l_V_orig
    l_V_sub
    
    z_orig    
    z_sub
    x_sub
    l_sub

    function SubModel(m::Model,V,rho;opt...)
        msub = Model(m.optimizer;m.opt...,opt...)

        V_bdry = Int[]

        V_bool = falses(num_variables(m))
        V_bool[V] .= true
        
        for i in 1:num_constraints(m)
            vs = keys(deriv(m.cons[i]))
            if examine(V_bool,vs) == 1
                union!(V_bdry,vs)
            end
        end

        sort!(V_bdry)

        V_inner = setdiff(V,V_bdry)
        V_con = [i for i in 1:num_constraints(m) if examine(V_bool,keys(deriv(m.cons[i]))) >= 1]
        
        x = Dict{Int,Expression}()
        z = Dict{Int,Expression}()
        l = Dict{Int,Expression}()
        
        for i in V_inner 
            x[i]= variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i])
        end
        for i in V_bdry
            x[i]= variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i])
            z[i]= parameter(msub)
        end
        for i in V_bdry
            l[i]= parameter(msub)
        end
        
        for obj in m.objs
            examine(V_bool,keys(deriv(obj))) >= 2 && objective(msub,func(obj)(x,m.p))
        end
        for i in V_con
             constraint(msub,func(m.cons[i])(x,m.p);lb=m.gl[i],ub=m.gu[i])
        end
        for i in V_bdry
            objective(msub, (x[i]-z[i]) * l[i])
            objective(msub, 0.5 * rho * (x[i]-z[i])^2)
        end
                
        instantiate!(msub)

        x_V_orig = view(m.x,V)
        x_V_sub = view(msub.x,([index(x[i]) for i in V]))
        l_V_orig = view(m.l,V_con)
        l_V_sub = msub.l

        z_orig = view(m[:z],V_bdry)
        z_sub = view(msub.p,1:length(V_bdry)) 
        x_sub = view(msub.x,length(V_inner)+1:length(V_inner)+length(V_bdry))
        l_sub = view(msub.p,1+length(V_bdry):2*length(V_bdry))

        return new(msub, x_V_orig, x_V_sub, l_V_orig, l_V_sub, z_orig, z_sub, x_sub, l_sub)
    end
end

mutable struct ADMMModel
    model::Model
    submodels::Vector{SubModel}
    rho::Float64
end

function iterate!(admm::ADMMModel;
                  err_pr=Threads.Atomic{Float64}(Inf),
                  err_du=Threads.Atomic{Float64}(Inf))
    Threads.@threads for sm in admm.submodels
        Threads.atomic_max!(err_du,admm.rho * difference(sm.z_orig,sm.z_sub))
        sm.z_sub .= sm.z_orig
        sm.l_sub .+= admm.rho .* (sm.x_sub .- sm.z_sub)
        optimize!(sm)
        sm.x_V_orig .= sm.x_V_sub
        Threads.atomic_max!(err_pr,difference(sm.x_sub,sm.z_sub))
    end
    admm.model.l.=0
    for sm in admm.submodels
        sm.l_V_orig .+= sm.l_V_sub
    end
    admm.model[:z] .= 0
    for sm in admm.submodels
        sm.z_orig .+= sm.x_sub + sm.l_sub ./ 2 ./ admm.rho
    end
    primal_update!(admm.model[:z],admm.model[:z_counter])    
end

function primal_update!(z,z_counter)
    @inbounds @simd for i in eachindex(z)
        z_counter[i] != 0 && (z[i] /= z_counter[i])
    end
end

function optimize!(admm::ADMMModel;tol = 1e-8, maxiter = 100, optional = (admm)->nothing)
    err_pr=Threads.Atomic{Float64}(Inf)
    err_du=Threads.Atomic{Float64}(Inf)

    iter = 0
    while max(err_pr[],err_du[]) > tol && iter < maxiter
        err_pr[] = .0
        err_du[] = .0
        iterate!(admm;err_pr=err_pr,err_du=err_du)
        println(@sprintf "%4i %4.2e %4.2e" iter+=1 err_pr[] err_du[])
        optional(admm)
    end
end

function difference(a,b)
    diff = .0
    @inbounds @simd for i in eachindex(a)
        diff = max(diff,abs(a[i] - b[i]))
    end
    return diff
end

function examine(W_bool,keys)
    numtrue = 0
    for k in keys
        W_bool[k] && (numtrue += 1)
    end
    return numtrue == 0 ? 0 : numtrue == length(keys) ? 2 : 1
end

optimize!(sm::SubModel) = optimize!(sm.model)
instantiate!(sm::SubModel) = instantiate!(sm.model)

function ADMMModel(m::Model;rho=1.,opt...)

    m[:z] = zeros(num_variables(m))
    m[:z_counter] = zeros(Int,num_variables(m))
    
    sms = Vector{SubModel}(undef,length(m[:Vs]))

    K = [k for k in keys(m[:Vs])]
    Threads.@threads for k in K
        sms[k] = SubModel(m,m[:Vs][k],rho;opt...)
    end

    for k in K
        m[:z_counter][sms[k].z_orig.indices[1]] .+= 1
    end
    
    ADMMModel(m,sms,rho)
end

end # module
