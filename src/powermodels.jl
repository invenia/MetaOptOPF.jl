abstract type AbstractOPFResult end

struct Primal <: AbstractOPFResult end
struct Dual <: AbstractOPFResult end
struct ActiveSet <: AbstractOPFResult end

abstract type Solver end

struct VerboseSolver <: Solver
    rand::Bool
end

VerboseSolver() = VerboseSolver(true)

function (s::VerboseSolver)(f::Function, output_file::AbstractString)
    solver = JuMP.with_optimizer(
        Ipopt.Optimizer,
        output_file = output_file,
        print_level = 0,
        file_print_level = 3
    )
    res = f(solver)
    ni, ct = solvestats(output_file)
    return res, ni, ct
end
function (s::VerboseSolver)(f::Function)
    workerid = string(myid())
    if s.rand
        res, ni, ct = mktempdir(;prefix="jl_" * workerid * "_") do dir
            output_file = joinpath(dir, "_solve.info")
            s(f, output_file)
        end
    else
        output_file = joinpath("./", "_solve.info")
        res, ni, ct = s(f, output_file)
    end
    return res, ni, ct
end


struct StandardSolver <: Solver end
function (s::StandardSolver)(f::Function)
    solver = JuMP.with_optimizer(
        Ipopt.Optimizer,
        print_level = 0,
    )
    result = f(solver)
end

function get_x(power_model::Dict)
    x = []
    for (i, load) in sort(power_model["load"])
        push!(x, load["pd"])
    end
    for (i, gen) in sort(power_model["gen"])
        push!(x, gen["pmax"])
    end
    for (i, branch) in sort(power_model["branch"])
        push!(x, branch["br_x"])
    end
    for (i, branch) in sort(power_model["branch"])
        push!(x, branch["rate_a"])
    end
    return x
end

"""
    function set_x!(power_model::Dict, case::Dict)
takes a PowerModels dictionary and changes its input parameters according to case
# Argument
- `power_model::Dict`: network structure in PowerModels dictionary format.
- `case::Dict`: sample in RunDCSampler format
"""
function set_x!(power_model::Dict, cases_row::Dict)
    k = 0
    for (i, load) in sort(power_model["load"])
        k += 1
        load["pd"] = cases_row["price_insensitive_load"][k]
    end
    k = 0
    for (i, gen) in sort(power_model["gen"])
        k += 1
        gen["pmax"] = cases_row["pg_max"][k]
    end
    k = 0
    for (i, branch) in sort(power_model["branch"])
        k += 1
        branch["br_x"] = cases_row["br_x"][k]
    end
    k = 0
    for (i, branch) in sort(power_model["branch"])
        k += 1
        branch["rate_a"] = cases_row["rate_a"][k]
    end
end

# for fixed structure using directly an input vector is faster
"""
    function set_x!(power_model::Dict, x::Vector)
takes a PowerModels dictionary and changes its input parameters according to `x` vector
# Argument
- `power_model::Dict`: network structure in PowerModels dictionary format.
- `x::Vector`: vector of parameter values to change
"""
function set_x!(data::Dict, x::Vector)
    k = 0
    for (i, load) in sort(data["load"])
        k += 1
        load["pd"] = x[k]
    end
    for (i, gen) in sort(data["gen"])
        k += 1
        gen["pmax"] = x[k]
    end
    for (i, branch) in sort(data["branch"])
        k += 1
        branch["br_x"] = x[k]
    end
    for (i, branch) in sort(data["branch"])
        k += 1
        branch["rate_a"] = x[k]
    end
end

"""
    function get_y(result::Dict)
takes the result of PowerModels run_dc_opf function and extracts primal optimizers' value
# Argument
- `power_model::Dict`: network structure in PowerModels dictionary format.
# Output
- `y`: vector of optimizers
"""
function get_y(result::Dict)
    y = []
    for (i, bus) in sort(result["bus"])
        append!(y, bus["va"])
    end
    for (i, gen) in sort(result["gen"])
        append!(y, gen["pg"])
    end
    for (i, branch) in sort(result["branch"])
        append!(y, branch["pf"])
    end
    return y
end

"""
    function set_y!(result::Dict, y::Vector)
takes a network in PowerModels format and sets its optimization variables according to y vector
# Argument
- `power_model::Dict`: network structure in PowerModels dictionary format.
- `y`: vector of optimization variables
"""
function set_y!(pm_model::Dict, y::Vector)
    k = 0
    for (i, bus) in sort(pm_model["bus"])
        k += 1
        bus["va_start"] = y[k]
    end
    for (i, gen) in sort(pm_model["gen"])
        k += 1
        gen["pg_start"] = y[k]
    end
    for (i, branch) in sort(pm_model["branch"])
        k += 1
        branch["pf_start"] = y[k]
    end
end

is_solved(x::MOI.TerminationStatusCode) = x == MOI.LOCALLY_SOLVED || x == MOI.OPTIMAL

function case_to_data(Cases::AbstractArray; t = Primal)
    OPF_data = []
    flag_case = []
    for j in 1:size(Cases, 1)
        if is_solved(Cases[j]["OPF_output"]["termination_status"])
            x = vcat(Cases[j]["price_insensitive_load"], Cases[j]["pg_max"], Cases[j]["br_x"], Cases[j]["rate_a"])
            y = target(t, Cases[j])
            append!(flag_case, j)
            push!(OPF_data, [x, y])
        end
    end
    return OPF_data, flag_case
end

target(::Type{Primal}, case) = get_y(case["OPF_output"]["solution"])
target(::Type{Dual}, case) = error("Not included?")
target(::Type{ActiveSet}, case) = case["OPF_regime"]

function generate_grid_data(network::Dict, par::Dict, N = 10, t = GoCompetition.ActiveSet)
    par["case_network"] = network
    cases = RunDCSampler(N, par);
    data, Flag_c = GoCompetition.case_to_data(cases; t = t)
    data_train, data_test = splitobs(data, 0.6)
end

function warmstart!(network::Dict, solution)
    for bus in keys(solution["bus"])
      network["bus"][bus]["va_start"] = solution["bus"][bus]["va"]
    end
    for branch in keys(solution["branch"])
      network["branch"][branch]["pf_start"] = solution["branch"][branch]["pf"]
    end
    for gen in keys(solution["gen"])
      network["gen"][gen]["pg_start"] = solution["gen"][gen]["pg"]
    end
end

function unbind!(network::Dict)
    for gen in keys(network["gen"])
        network["gen"][gen]["pmin"] = -999
        network["gen"][gen]["pmax"] = 999
    end
    for branch in keys(network["branch"])
        network["branch"][branch]["rate_a"] = 9999
    end
end
