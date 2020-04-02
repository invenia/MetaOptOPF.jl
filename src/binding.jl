export get_binding_status, get_violation_status
export remove_nonbinding_constraints!, extend_binding_status!, update_binding_status!
export verify_binding_status, verify_violation_status



AC_REL_CONSTRAINT_TYPES = [
    (VariableRef, MathOptInterface.GreaterThan{Float64})
    (VariableRef, MathOptInterface.LessThan{Float64})
    (GenericAffExpr{Float64,VariableRef}, MathOptInterface.GreaterThan{Float64})
    (GenericAffExpr{Float64,VariableRef}, MathOptInterface.LessThan{Float64})
    #(GenericAffExpr{Float64,VariableRef}, MathOptInterface.EqualTo{Float64})
    (GenericQuadExpr{Float64,VariableRef}, MathOptInterface.LessThan{Float64})
    #(GenericQuadExpr{Float64,VariableRef}, MathOptInterface.EqualTo{Float64})
]

"""
    get_binding_status(model::Model; threshold::Float64=1.0e-5)

Returns a Dict with constraint type keys and an array of bool values indicating wether the
corresponding constraint is binding/active (true) or non-binding/inactive (false).
# Arguments
- `model::Model`: solved OPF JuMP model
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.

# Returns
- `Dict`: Dictionary with constraint type keys and array of bool values.
"""
function get_binding_status(model::Model; threshold::Float64=1.0e-5)
    termination_status(model) == OPTIMIZE_NOT_CALLED && error("model is not solved!")
    binding_status = Dict()
    for constraint_type in list_of_constraint_types(model)
        binding_status[constraint_type] = ones(Bool, num_constraints(model, constraint_type...))
        if constraint_type[2] == MOI.EqualTo{Float64}
            continue  # equality constraints are always binding
        else
            c = all_constraints(model, constraint_type...)
            s = map(ci -> constraint_object(ci).set, c)
            lhs = value.(c)
            rhs = map(si -> getproperty(si, fieldnames(typeof(si))[1]), s)
            diff = lhs - rhs
            if constraint_type[2] == MOI.GreaterThan{Float64}
                diff = map(d -> d < 0.0 ? 0.0 : abs(d), diff)
            else
                diff = map(d -> d > 0.0 ? 0.0 : abs(d), diff)
            end
            binding_status[constraint_type][diff .> threshold] .= false
        end
    end
    return binding_status
end

function get_binding_status(model_left::Model, model_right::Model; threshold::Float64=1.0e-5)
    termination_status(model_left) == OPTIMIZE_NOT_CALLED && error("model_left is not solved!")
    binding_status = Dict()
    for constraint_type in list_of_constraint_types(model_left)
        binding_status[constraint_type] = ones(Bool, num_constraints(model_left, constraint_type...))
        if constraint_type[2] == MOI.EqualTo{Float64}
            continue  # equality constraints are always binding
        else
            for i=1:num_constraints(model_left, constraint_type...)
                c_left = all_constraints(model_left, constraint_type...)[i]
                c_right = all_constraints(model_right, constraint_type...)[i]
                s_right = constraint_object(c_right).set
                lhs = value(c_left)
                rhs = getproperty(s_right, fieldnames(typeof(s_right))[1])
                diff = if constraint_type[2] == MOI.GreaterThan{Float64}
                    lhs < rhs ? 0.0 : abs(lhs - rhs)  # violation is considered as binding
                else
                    lhs > rhs ? 0.0 : abs(lhs - rhs)  # violation is considered as binding
                end
                if diff > threshold
                    binding_status[constraint_type][i] = false
                end
            end
        end
    end
    return binding_status
end

"""
    get_binding_status(power_model::GenericPowerModel; threshold::Float64=1.0e-5)

Returns a Dict with constraint type keys and an array of bool values indicating wether the
corresponding constraint is binding/active (true) or non-binding/inactive (false).
# Arguments
- `power_model::GenericPowerModel`: solved OPF PowerModels model
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.

# Returns
- `Dict`: Dictionary with constraint type keys and array of bool values.
"""
function get_binding_status(power_model::GenericPowerModel; threshold::Float64=1.0e-5)
    return get_binding_status(power_model.model, threshold = threshold)
end

function get_binding_status(power_model_left::GenericPowerModel, power_model_right::GenericPowerModel; threshold::Float64=1.0e-5)
    return get_binding_status(power_model_left.model, power_model_right.model, threshold = threshold)
end

"""
    get_violation_status(model::Model; threshold::Float64=1.0e-5)

Returns a Dict with constraint type keys and an array of bool values indicating wether the
corresponding constraint is violated (true) or not (false).
# Arguments
- `model::Model`: solved OPF JuMP model
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.

# Returns
- `Dict`: Dictionary with constraint type keys and array of bool values.
"""
function get_violation_status(model::Model; threshold::Float64=1.0e-5)
    termination_status(model) == OPTIMIZE_NOT_CALLED && error("model is not solved!")
    violation_status = Dict()
    for constraint_type in list_of_constraint_types(model)
        violation_status[constraint_type] = zeros(Bool, num_constraints(model, constraint_type...))
        if constraint_type[2] == MOI.EqualTo{Float64}
            continue  # equality constraints are always present so never violated
        else
            for i=1:num_constraints(model, constraint_type...)
                c = all_constraints(model, constraint_type...)[i]
                s = constraint_object(c).set
                lhs = value(c)
                rhs = getproperty(s, fieldnames(typeof(s))[1])
                diff = if constraint_type[2] == MOI.GreaterThan{Float64}
                    lhs > rhs ? 0.0 : abs(lhs - rhs)
                else
                    lhs < rhs ? 0.0 : abs(lhs - rhs)
                end
                if diff > threshold
                    violation_status[constraint_type][i] = true
                end
            end
        end
    end
    return violation_status
end

"""
    get_violation_status(power_model::GenericPowerModel; threshold::Float64=1.0e-5)

Returns a Dict with constraint type keys and an array of bool values indicating wether the
corresponding constraint is violated (true) or not (false).
# Arguments
- `power_model::GenericPowerModel`: solved OPF PowerModels model
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.

# Returns
- `Dict`: Dictionary with constraint type keys and array of bool values.
"""
function get_violation_status(power_model::GenericPowerModel; threshold::Float64=1.0e-5)
    return get_violation_status(power_model.model, threshold=threshold)
end

"""
    remove_nonbinding_constraints!(model::Model, binding_status::Dict)

Removes all non-binding constraints from JuMP model based on binding_status.
# Arguments
- `model::GenericPowerModel`: full JuMP model
- `binding_status::Dict`: dictionary including the binding status of each constraints

# Returns
- `n`: Number of inactive and removed constraints.
"""
function remove_nonbinding_constraints!(model::Model, binding_status::Dict)
    n = 0
    for constraint_type in keys(binding_status)
        length(binding_status[constraint_type]) != num_constraints(model, constraint_type...) && error("model is not full!")
        constraint_type[2] == MOI.EqualTo{Float64} && continue  # equality constraints are always binding
        n += length(delete.(model, all_constraints(model, constraint_type...)[.!binding_status[constraint_type]]))
    end
    return n
end

"""
    remove_nonbinding_constraints!(power_model::GenericPowerModel, binding_status::Dict)

Removes all non-binding constraints from PowerModels model based on binding_status.
# Arguments
- `power_model::GenericPowerModel`: full PowerModels model
- `binding_status::Dict`: dictionary including the binding status of each constraints

# Returns
- `n`: Number of inactive and removed constraints.
"""
function remove_nonbinding_constraints!(power_model::GenericPowerModel, binding_status::Dict)
    return remove_nonbinding_constraints!(power_model.model, binding_status)
end

"""
    extend_binding_status!(network::Dict{String,Any}, reduced_opf_result::Dict, binding_status::Dict; threshold=1.0e-5)

Extends the binding_status with active constraints based on the solution of reduced OPF.
A full PowerModels network is also required.
# Arguments
- `network::Dict{String,Any}`: full grid PowerModels network
- `reduced_opf_result::Dict`: results of the reduced model returned by PowerModels
- `binding_status::Dict`: dictionary including the binding status of each constraints
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.

# Returns
- `n`: Number of updated constraints
"""
function extend_binding_status!(network::Dict{String,Any}, reduced_opf_result::Dict, binding_status::Dict; threshold=1.0e-5)

    new_binding_status = verify_binding_status(network, reduced_opf_result["solution"], threshold)
    n = 0
    for constraint_type in keys(binding_status)
        constraint_type[2] == MOI.EqualTo{Float64} && continue
        for i=1:length(binding_status[constraint_type])
            if (!binding_status[constraint_type][i]) && new_binding_status[constraint_type][i]
                binding_status[constraint_type][i] = new_binding_status[constraint_type][i]
                n += 1
            end
        end
    end
    return n
end

"""
    update_binding_status!(network::Dict{String,Any}, reduced_opf_result::Dict, binding_status::Dict; threshold=1.0e-5)

Updates the binding_status based on the solution of reduced OPF.
A full PowerModels network is also required.
# Arguments
- `network::Dict{String,Any}`: full grid PowerModels network
- `reduced_opf_result::Dict`: results of the reduced model returned by PowerModels
- `binding_status::Dict`: dictionary including the binding status of each constraints
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.

# Returns
- `n`: Number of updated constraints
"""
function update_binding_status!(network::Dict{String,Any}, reduced_opf_result::Dict, binding_status::Dict; threshold=1.0e-5)

    new_binding_status = verify_binding_status(network, reduced_opf_result["solution"], threshold)

    n = 0
    for constraint_type in keys(binding_status)
        constraint_type[2] == MOI.EqualTo{Float64} && continue
        for i=1:length(binding_status[constraint_type])
            if binding_status[constraint_type][i] != new_binding_status[constraint_type][i]
                binding_status[constraint_type][i] = new_binding_status[constraint_type][i]
                n += 1
            end
        end
    end
    return n
end

function verify_binding_status(network::Dict, solution, threshold)
    _network = deepcopy(network)
    warmstart!(_network, solution)
    full_power_model_right = build_model(_network, ACPPowerModel, PowerModels.post_opf)
#   full_opf_result_right = optimize_model!(full_power_model_right, JuMP.with_optimizer(Ipopt.Optimizer, max_iter=0, print_level=0))
    unbind!(_network)
    full_power_model_left = build_model(_network, ACPPowerModel, PowerModels.post_opf)
    full_opf_result_left = optimize_model!(full_power_model_left, JuMP.with_optimizer(Ipopt.Optimizer, max_iter=0, print_level=0))
    return get_binding_status(full_power_model_left, full_power_model_right, threshold=threshold)
end

function verify_violation_status(network::Dict, solution, threshold)
    _network = deepcopy(network)
    warmstart!(_network, solution)
    full_power_model = build_model(_network, ACPPowerModel, PowerModels.post_opf)
    full_opf_result = optimize_model!(full_power_model, JuMP.with_optimizer(Ipopt.Optimizer, max_iter=0, print_level=0))
    return get_violation_status(full_power_model, threshold=threshold)
end

relconstrainttypes(m) = filter(x -> x in AC_REL_CONSTRAINT_TYPES, list_of_constraint_types(m))
totalconstraints(m) = sum(map(relconstrainttypes(m)) do (x) num_constraints(m, x...) end)
ntconstraints(m) = map(relconstrainttypes(m)) do (x); (ct=(x[1], x[2]), N=num_constraints(m, x...)) end

function classifier2binding!(b::Dict, c::T; n_gen=0, n_bus=0)  where {T<:AbstractArray{Bool}}
    k = 0
    for constraint_type in GoCompetition.AC_REL_CONSTRAINT_TYPES
        b[constraint_type][:] = c[k+1:k+length(b[constraint_type])]
        if constraint_type == AC_REL_CONSTRAINT_TYPES[1]
            b[constraint_type][1:n_bus+n_gen] .= true
        elseif constraint_type == AC_REL_CONSTRAINT_TYPES[2]
            b[constraint_type][1:n_bus] .= true
        end
        k += length(b[constraint_type])
    end
end

binding2classifier(b) = vcat(map(AC_REL_CONSTRAINT_TYPES) do (t) b[t] end ...)
create_bindingdict_template(data::Dict) = deepcopy(data)
