module SimpleADMM

import Printf: @sprintf
import LightGraphs: Graph, add_edge!, neighbors, nv
import SimpleNLModels: Expression, Model, variable, parameter, constraint, objective,
    num_variables, num_constraints, func, deriv, optimize!, instantiate!

export ADMMModel, iterate!, optimize!

mutable struct SubModel
    model::Model

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

        V_inner = setdiff(V,V_bdry)


        x = Dict{Int,Expression}()
        z = Dict{Int,Expression}()
        l = Dict{Int,Expression}()
        
        for i in V_inner 
            x[i]= variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i])
        end
        for i in V_bdry
            x[i]= variable(msub;lb=m.xl[i],ub=m.xu[i],start=m.x[i])
            z[i]= parameter(msub)
            l[i]= parameter(msub)
        end

        for obj in m.objs
            examine(V_bool,keys(deriv(obj))) >= 2 && objective(msub,func(obj)(x))
        end
        for i in eachindex(m.cons)
            V_bool[minimum(keys(deriv(m.cons[i])))] && constraint(msub,func(m.cons[i])(x);lb=m.gl[i],ub=m.gu[i])
        end
        for i in V_bdry
            objective(msub, (x[i]-z[i]) * l[i] + 0.5 * rho * (x[i]-z[i])^2 )
        end
                
        instantiate!(msub)
        
        z_orig = view(m[:z],V_bdry)
        z_sub = view(msub.p,1:length(V_bdry)) 
        x_sub = view(msub.x,length(V_inner)+1:length(V_inner)+length(V_bdry))
        l_sub = view(msub.p,1+length(V_bdry):2*length(V_bdry))

        m[:z_counter][V_bdry] .+= 1

        return new(msub, z_orig, z_sub, x_sub, l_sub)
    end
end

mutable struct ADMMModel
    model::Model
    submodels::Vector{SubModel}
    rho::Float64
end


function instantiate!(schwarz::ADMMModel)
    Threads.@threads for sm in schwarz.submodels
        instantiate!(sm)
    end
end
function iterate!(schwarz::ADMMModel;err=Threads.Atomic{Float64}(Inf))
    # Threads.@threads
    for sm in schwarz.submodels
        set_submodel!(sm,schwarz.rho)
        optimize!(sm)
        set_err!(sm,err)
    end
    schwarz.model[:z].=0
    for sm in schwarz.submodels
        set_orig!(sm)
    end
    primal_update!(schwarz.model[:z],schwarz.model[:z_counter])
end

function primal_update!(z,z_counter)
    @inbounds @simd for i in eachindex(z)
        z_counter[i] != 0 && (z[i] /= z_counter[i])
    end
end

function optimize!(schwarz::ADMMModel;tol = 1e-8, maxiter = 100)
    err=Threads.Atomic{Float64}(Inf)

    iter = 0
    while err[] > tol && iter < maxiter
        err[] = .0
        iterate!(schwarz;err=err)
        println(@sprintf "%4i %4.2e" iter+=1 err[])
    end
end

function set_err!(sm,err)
    Threads.atomic_max!(err,difference(sm.x_sub,sm.z_sub))
end

function difference(a,b)
    diff = .0
    @inbounds @simd for i in eachindex(a)
        diff = max(diff,abs(a[i] - b[i]))
    end
    return diff
end

function set_submodel!(sm,rho)
    sm.z_sub .= sm.z_orig
    sm.l_sub .+= rho .* (sm.x_sub .- sm.z_orig) 
    nothing
end

function set_orig!(sm)
    sm.z_orig .+= sm.x_sub
    nothing
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
    
    Threads.@threads for i in collect(keys(m[:Vs]))
        sms[i] = SubModel(m,m[:Vs][i],rho;opt...)
    end
    
    ADMMModel(m,sms,rho)
end

end # module
