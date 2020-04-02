@testset "Binding functions" begin

    setting = Dict("output" => Dict("branch_flows" => true))
    solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)
    nosolver = JuMP.with_optimizer(Ipopt.Optimizer, max_iter=0, print_level=0)

    network = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")

    full_pm = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    full_opf_result = optimize_model!(full_pm, solver)

    binding_status = get_binding_status(full_pm)
    @test typeof(binding_status) == Dict{Any, Any}
    @test length(keys(binding_status)) == 5

    function is_there_violation(violation_status)
        for key in keys(violation_status)
            any(violation_status[key]) && return true
        end
        return false
    end
    violation_status = get_violation_status(full_pm)
    @test typeof(violation_status) == Dict{Any, Any}
    @test length(keys(violation_status)) == 5
    @test !is_there_violation(violation_status)

    reduced_pm_1 = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    binding_status_1 = deepcopy(binding_status)
    @test remove_nonbinding_constraints!(reduced_pm_1, binding_status_1) == 167
    reduced_opf_result_1 = optimize_model!(reduced_pm_1, solver)
    @test !is_there_violation(verify_violation_status(deepcopy(network), reduced_opf_result_1["solution"], 1.0e-5))
    @test update_binding_status!(deepcopy(network), reduced_opf_result_1, binding_status_1) == 0

    reduced_pm_2 = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    binding_status_2 = deepcopy(binding_status)
    binding_status_2[(VariableRef, MOI.GreaterThan{Float64})] .= false
    binding_status_2[(VariableRef, MOI.LessThan{Float64})] .= false
    @test remove_nonbinding_constraints!(reduced_pm_2, binding_status_2) == 176
    reduced_opf_result_2 = optimize_model!(reduced_pm_2, solver)
    @test is_there_violation(verify_violation_status(deepcopy(network), reduced_opf_result_2["solution"], 1.0e-5))
    @test update_binding_status!(deepcopy(network), reduced_opf_result_2, binding_status_2) >= 45

    reduced_pm_3 = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    binding_status_3 = deepcopy(binding_status)
    binding_status_3[(VariableRef, MOI.GreaterThan{Float64})] .= false
    binding_status_3[(VariableRef, MOI.LessThan{Float64})] .= false
    @test remove_nonbinding_constraints!(reduced_pm_3, binding_status_3) == 176
    reduced_opf_result_3 = optimize_model!(reduced_pm_3, solver)
    @test extend_binding_status!(deepcopy(network), reduced_opf_result_3, binding_status_3) >= 45

    # Test behaves appropriately out of bounds
    network = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")
    pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    res = optimize_model!(pm_model, solver)
    b_solved = GoCompetition.get_binding_status(pm_model)
    ct_solved = list_of_constraint_types(pm_model.model)

    # Reload the network and pm_model
    network = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")
    pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)

    # Now fix them so that constraints should be violated
    # This will make the problem infeasible.
    map(all_variables(pm_model.model)) do (v) fix(v, rand(), force=true) end
    res = optimize_model!(pm_model, nosolver)
    @test !GoCompetition.is_solved(res["termination_status"])

    b_unsolved = GoCompetition.get_binding_status(pm_model)
    ct_unsolved = list_of_constraint_types(pm_model.model)

    @test_broken ct_solved == ct_unsolved # This is OK, because fixing removed generator upper/lower bounds.
    tunion = collect(intersect(keys(b_solved), keys(b_unsolved))) # Get the intersect
    @test any(map(tunion) do (ct) b_solved[ct] != b_unsolved[ct] end)

    # Filter relevant constraints
    @test length(GoCompetition.relconstrainttypes(full_pm.model)) == 4 # Two DC constraint types
    @test GoCompetition.totalconstraints(full_pm.model) ==
        num_constraints(full_pm.model, VariableRef, MathOptInterface.GreaterThan{Float64}) +
        num_constraints(full_pm.model, VariableRef, MathOptInterface.LessThan{Float64}) +
        num_constraints(full_pm.model, GenericAffExpr{Float64,VariableRef}, MathOptInterface.GreaterThan{Float64}) +
        num_constraints(full_pm.model, GenericAffExpr{Float64,VariableRef}, MathOptInterface.LessThan{Float64})

    @test GoCompetition.ntconstraints(full_pm.model) ==
        [
        (ct = (GenericAffExpr{Float64,VariableRef}, MathOptInterface.GreaterThan{Float64}), N = 41)
        (ct = (GenericAffExpr{Float64,VariableRef}, MathOptInterface.LessThan{Float64}), N = 41)
        (ct = (VariableRef, MathOptInterface.GreaterThan{Float64}), N = 47)
        (ct = (VariableRef, MathOptInterface.LessThan{Float64}), N = 47)
        ]

end
