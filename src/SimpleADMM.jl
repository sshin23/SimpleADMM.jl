module SimpleADMM

import Printf: @sprintf
import SimpleNL: Expression, Model, variable, parameter, constraint, objective, num_variables, num_constraints, optimize!, instantiate!, index,  non_caching_eval
import SimpleNLUtils: get_terms, get_entries_expr, sparsity, KKTErrorEvaluator
import Requires: @require

default_subproblem_optimizer() = @isdefined(DEFAULT_SUBPROBLEM_OPTIMIZER) ? DEFAULT_SUBPROBLEM_OPTIMIZER : error("DEFAULT_SUBPROBLEM_OPTIMIZER is not defined. To use Ipopt as a default subproblem optimizer, do: using Ipopt")
default_option() = Dict(
    :rho=>1.,
    :maxiter=>400,
    :tol=>1e-6,
    :subproblem_optimizer=>default_subproblem_optimizer(),
    :subproblem_option=>Dict(:print_level=>0),
    :save_output=>false,
    :optional=>admm->nothing
)

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

    function SubModel(m::Model,optimizer,V,rho,
                      objs,cons,obj_sparsity,con_sparsity;opt...)
        msub = Model(optimizer;opt...)

        V_bdry = Int[]

        V_bool = falses(num_variables(m))
        V_bool[V] .= true
        
        for vs in con_sparsity
            if examine(V_bool,vs) == 1
                union!(V_bdry,vs)
            end
        end

        sort!(V_bdry)

        V_inner = setdiff(V,V_bdry)
        V_con = [i for i in eachindex(con_sparsity) if examine(V_bool,con_sparsity[i]) >= 1]
        
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
        for i in V_con
             constraint(msub,non_caching_eval(cons[i],x,m.p);lb=m.gl[i],ub=m.gu[i])
        end
        
        objective(msub,sum(non_caching_eval(objs[i],x,m.p) for i in eachindex(objs) if examine(V_bool,obj_sparsity[i]) >= 2) + sum((x[i]-z[i]) * l[i] + 0.5 * rho * (x[i]-z[i])^2 for i in V_bdry))
                
        instantiate!(msub)

        x_V_orig = view(m.x,V)
        x_V_sub = view(msub.x,([x[i].index for i in V]))
        l_V_orig = view(m.l,V_con)
        l_V_sub = msub.l

        z_orig = view(m[:z],V_bdry)
        z_sub = view(msub.p,1:length(V_bdry)) 
        x_sub = view(msub.x,length(V_inner)+1:length(V_inner)+length(V_bdry))
        l_sub = view(msub.p,1+length(V_bdry):2*length(V_bdry))

        return new(msub, x_V_orig, x_V_sub, l_V_orig, l_V_sub, z_orig, z_sub, x_sub, l_sub)
    end
end

mutable struct Optimizer
    model::Model
    submodels::Vector{SubModel}
    kkt_error_evaluator
    opt::Dict{Symbol,Any}
end

function primal_update!(z,z_counter)
    @inbounds @simd for i in eachindex(z)
        z_counter[i] != 0 && (z[i] /= z_counter[i])
    end
end

function optimize!(admm::Optimizer)
    save_output = admm.opt[:save_output]
    if save_output
        output = Tuple{Float64,Float64}[]
        start = time()
    end

    iter = 0
    while (err=admm.kkt_error_evaluator(admm.model.x,admm.model.l,admm.model.gl)) > admm.opt[:tol] && iter < admm.opt[:maxiter]
        save_output && push!(output,(err,time()-start))
        println(@sprintf "%4i %4.2e" iter+=1 err)

        Threads.@threads for sm in admm.submodels
            sm.z_sub .= sm.z_orig
            sm.l_sub .+= admm.opt[:rho] .* (sm.x_sub .- sm.z_sub)
            optimize!(sm)
            sm.x_V_orig .= sm.x_V_sub
        end
        admm.model.l.=0
        for sm in admm.submodels
            sm.l_V_orig .+= sm.l_V_sub
        end
        admm.model[:z] .= 0
        for sm in admm.submodels
            sm.z_orig .+= sm.x_sub + sm.l_sub ./ 2 ./ admm.opt[:rho]
        end
        primal_update!(admm.model[:z],admm.model[:z_counter])    
    end
    save_output && push!(output,(err,time()-start))
    println(@sprintf "%4i %4.2e" iter+=1 err)
    save_output && (admm.model.ext[:output]=output)
    admm.opt[:optional](admm)
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

function Optimizer(m::Model)

    opt = default_option()
    for (sym,val) in m.opt
        opt[sym] = val
    end

    objs = get_terms(m.obj)
    cons = get_entries_expr(m.con)
    obj_sparsity = [sparsity(e) for e in objs]
    con_sparsity = [sparsity(e) for e in cons]

    m[:z] = zeros(num_variables(m))
    m[:z_counter] = zeros(Int,num_variables(m))
    
    sms = Vector{SubModel}(undef,length(m[:Vs]))

    K = [k for k in keys(m[:Vs])]
    Threads.@threads for k in K
        sms[k] = SubModel(m,opt[:subproblem_optimizer],m[:Vs][k],opt[:rho],
                          objs,cons,obj_sparsity,con_sparsity;
                          opt[:subproblem_option]...)
    end

    for k in K
        m[:z_counter][sms[k].z_orig.indices[1]] .+= 1
    end
    
    Optimizer(m,sms,KKTErrorEvaluator(m),opt)
end

function __init__()
    @require Ipopt="b6b21f68-93f8-5de0-b562-5493be1d77c9" @eval begin
        import ..Ipopt
        DEFAULT_SUBPROBLEM_OPTIMIZER = Ipopt.Optimizer
    end
end

end # module
