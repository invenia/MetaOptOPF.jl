@testset "PowerModels" begin

  pm_model = PowerModels.parse_file("./data/case9.m");
  pm_model_deepcopy = deepcopy(pm_model);

  par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
      "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);
  cases = RunDCSampler(3, par);

  z_star = GoCompetition.get_y(cases[1]["OPF_output"]["solution"]);
  GoCompetition.set_y!(pm_model, z_star)
  GoCompetition.set_y!(pm_model_deepcopy, z_star)

  # These should be the same
  @test pm_model["bus"]["8"] == pm_model_deepcopy["bus"]["8"]

  pm_model = PowerModels.parse_file("./data/case9.m");
  pm_model_deepcopy = deepcopy(pm_model);

  par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
      "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);
  cases = RunDCSampler(3, par);

  opf_result = run_dc_opf(pm_model, JuMP.with_optimizer(Ipopt.Optimizer, print_level=0))
  z_star = GoCompetition.get_y(cases[1]["OPF_output"]["solution"])

  # Test initialization warm_start
  GoCompetition.set_x!(pm_model, cases[1])
  # Test the various values have been updated
  for (key, val) in pm_model_deepcopy["load"]
    @test val["pd"] != pm_model["load"][key]["pd"]
  end
  for (key, val) in pm_model_deepcopy["gen"]
    @test val["pmax"] != pm_model["gen"][key]["pmax"]
  end
  for (key, val) in pm_model_deepcopy["branch"]
    @test val["br_x"] != pm_model["branch"][key]["br_x"]
  end
  for (key, val) in pm_model_deepcopy["branch"]
    @test val["rate_a"] != pm_model["branch"][key]["rate_a"]
  end
  # Test no unexpected changes (TODO: Make generic)
  for (key, val) in pm_model_deepcopy["gen"]
    @test val["startup"] == pm_model["gen"][key]["startup"]
  end

  # Test extracting primals correctly
  # Create a test dict, deliberately out of order w.r.t ID.
  # We do a sort here so that order is preserved w.r.t value insertion.
  # (Otherwise, the actual Dict could change order, making the test invalid
  # In the future we should avoid needing to resort the Dictionary

  r = rand(1,)[1]
  td =  Dict(
    "bus" => sort(Dict(("2" => Dict("va" => 0.5), "1" => Dict("va" => -0.2)))),
    "gen" => sort(Dict(("1" => Dict("pg" => 50.), "2" => Dict("pg" => r)))),
    "branch" => sort(Dict("3" => Dict("pf" => 1e-2), "2" => Dict("pf" => -1e-3))),
    )

  # This iterates through a dict and adds to a vector, so the main thing to check is that
  # a) the values are correct and b) that re-ordering is handled
  @test GoCompetition.get_y(td) == [-0.2, 0.5, 50., r, -1e-3, 1e-2]

  # TODO: An important test to write here is about Order Preservation between the set and get methods
  # As they are vectors not Dicts.
  N = length(z_star)
  y = rand(N, )

  GoCompetition.set_y!(pm_model_deepcopy, y)
  let pm = pm_model_deepcopy
    k = 0; (inc() = k += 1)
    for (key, val) in sort(pm["bus"])
        inc(); @test val["va_start"] == y[k]
    end
    for (key, val) in sort(pm["gen"])
        inc(); @test val["pg_start"] == y[k]
    end
    for (key, val) in sort(pm["branch"])
        inc(); @test val["pf_start"] == y[k]
    end
  end

  # pm_model = PowerModels.parse_file("./data/case9.m")
  # par = Dict("case_network" => pm_model, "dev_load_pd" => 0.,
  #    "dev_gen_max" => 0., "dev_rate_a" => 0., "dev_br_x" => 0.)
  # cases = RunSampler(3, par)

end

@testset "Set and g w" begin

    size_in = 10
    size_out = 4
    model1 = Chain(Dense(size_in, 50), BatchNorm(50, relu), Dropout(0.4),
                   Dense(50, 50), BatchNorm(50, relu), Dropout(0.4),
                   Dense(50, size_out, σ))

    model2 = Chain(Dense(size_in, 50), BatchNorm(50, relu), Dropout(0.4),
                  Dense(50, 50), BatchNorm(50, relu), Dropout(0.4),
                  Dense(50, size_out, σ))

    r = rand(size_in, 7)
    # Enable testmodel to disable dropout (stochastic)
    Flux.testmode!(model1)
    Flux.testmode!(model2)
    @test model1(r) != model2(r) # Different initializations

    _w = GoCompetition.get_w(model1)
    GoCompetition.set_w!(model2, _w)

    @test model1(r) == model2(r) # Different initializations

    # Note on BatchNorm parameters:
    # https://github.com/FluxML/Flux.jl/issues/492
    # It works here because it initializes to 0 and 1

end


@testset "Model tests" begin
  # Generate a pm_model
  pm_model = PowerModels.parse_file("./data/case300.m")
  par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
      "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1)
  cases = RunDCSampler(10, par)

  data, Flag_c = GoCompetition.case_to_data(cases)
  data_train, data_test = splitobs(data, 0.6)

  N_in = size(data[1][1], 1)
  N_out = size(data[1][2], 1)

  # Define optimization routine
  model = GoCompetition.NNmodel(N_in, N_out)
  opt = ADAM(0.01)
  loss(x, y) = Flux.mse(model(x), y)
  Flux.@epochs 5 Flux.train!(loss, Flux.params(model), data_train, opt)
  solve_time_std, solve_time_exact, solve_time_model =
    GoCompetition.evaluate(1, pm_model, model, data_test)
  (μ1, σ1), (μ2, σ2) = GoCompetition.summarise(solve_time_std, solve_time_exact, solve_time_model)

  # For now, we are not expecting full initialization (primals initialized but not duals)
  @test_broken Statistics.std(solve_time_exact[1, :]) == 0.0

  # We expect that most of the time, the average exact solve is faster than the
  # average standard solve (using default primal initialization)
  S = length(data_train)
  primal_init_gain = (1 / S) *
    (sum(Statistics.mean(solve_time_exact; dims=2) .< Statistics.mean(solve_time_std; dims=2)))
  #@test primal_init_gain >= 0.6 # Commented out in favour of testing gain is positive

  # Test that there is real gain when using the exact solution.
  # Do not test the model gain at this point (as not really trained)
  @test μ1 > 0

end

@testset "Metaloss" begin
  pm_model = PowerModels.parse_file("./data/case9.m")
  par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
      "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1)
  cases = RunDCSampler(5, par)

  data, Flag_c = GoCompetition.case_to_data(cases)
  data_train, data_test = splitobs(data, 0.6)

  N_in = size(data[1][1], 1)
  N_out = size(data[1][2], 1)

  # Define Model 1
  model = GoCompetition.NNmodel(N_in, N_out)
  opt = ADAM(0.01)
  loss(x, y) = Flux.mse(model(x), y)
  Flux.@epochs 5 Flux.train!(loss, Flux.params(model), data_train, opt)

  # Define Model 2
  metamodel = GoCompetition.NNmodel(N_in, N_out)
  metaloss(w) = GoCompetition.metaloss(w, metamodel, pm_model, data_train;
    solver = VerboseSolver)

    @testset "Solver Stats" begin
      # Test that solver statistics can be extracted
      metaloss(w) = GoCompetition.metaloss(w, metamodel, pm_model, data_train;
        solver = VerboseSolver(false))
      metaloss(GoCompetition.get_w(metamodel))
      Niter0, ctime0 = GoCompetition.solvestats("./_solve.info")

      # Test that this is different with a re-run/different model
      w = rand(Float64, size(GoCompetition.get_w(metamodel)))
      metaloss(w)
      Niter1, ctime1 = GoCompetition.solvestats("./_solve.info")

      # This could fail stochastically due to rounding of the output of time
      @test (ctime0 != ctime1 || Niter0 != Niter1)

    end

    @testset "Running PSO" begin

      Niter = 10
      options = Optim.Options(iterations=Niter, show_trace=false)
      res = optimize(metaloss, GoCompetition.get_w(metamodel), ParticleSwarm(n_particles=5, all_from_init=true), options)

      #  Test no side effects before setting value
      w_init = GoCompetition.get_w(metamodel)

      # Test that the particle swarm has found new values.
      @test w_init != res.minimizer

      # Test that this was able to run
      @test res.iterations == Niter

      # Test that some minimization has in fact occured, explicitely setting the metamodel to the min
      GoCompetition.set_w!(metamodel, res.minimizer) # set the metamodel to the soloution of PSO
      @test metaloss(GoCompetition.get_w(metamodel)) <= metaloss(w_init)

      solve_time_std, solve_time_exact, solve_time_model, solve_time_metamodel =
        GoCompetition.evaluate(10, pm_model, model, metamodel, data_test)
      (μ1, σ1), (μ2, σ2), (μ3, σ3) = GoCompetition.summarise(
        solve_time_std, solve_time_exact, solve_time_model, solve_time_metamodel)

      # Test that these are real values. We don't test performance at this point
      # as deliberately very few examples.
      @test μ1 isa Real
      @test μ2 isa Real
      @test μ3 isa Real
      @test σ1 isa Real
      @test σ2 isa Real
      @test σ3 isa Real

    end

end

@testset "Reduced OPF Pipeline" begin

    setting = Dict("output" => Dict("branch_flows" => true))
    solver = JuMP.with_optimizer(Ipopt.Optimizer, print_level=0)
    network = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")

    full_pm = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    full_opf_result = optimize_model!(full_pm, solver)

    binding_status = get_binding_status(full_pm)

    obj = GoCompetition.objective_extend!(deepcopy(network), deepcopy(binding_status))
    @test obj[:ctime] < obj[:stime]

    @test GoCompetition.objective_extend!(deepcopy(network), deepcopy(binding_status))[:niter] == 6
    @test GoCompetition.objective_update!(deepcopy(network), deepcopy(binding_status))[:niter] == 6

    binding_status[(VariableRef, MOI.LessThan{Float64})] .= false
    binding_status[(VariableRef, MOI.GreaterThan{Float64})] .= false

    @test GoCompetition.objective_extend!(deepcopy(network), deepcopy(binding_status))[:niter] == 56
    @test GoCompetition.objective_update!(deepcopy(network), deepcopy(binding_status))[:niter] >= 52

end

@testset "Active Set Classification" begin
# Build a Classifier Network
    network = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")
    setting = Dict("output" => Dict("branch_flows" => true))
    full_pm = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)

    c = convert.(Bool, round.(rand(GoCompetition.totalconstraints(full_pm.model))))
    optimize_model!(full_pm, GoCompetition.ipopt_solver) # Need this to get constraints dict
    b = get_binding_status(full_pm)
    b = GoCompetition.create_bindingdict_template(b) # Actually unncessary but it makes clear
    GoCompetition.classifier2binding!(b, c)

    # Test the classifier binding functions
    @test GoCompetition.binding2classifier(b) == c

    # Build a model
    par = Dict("case_network" => network, "dev_load_pd" => 0.1,
        "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1)
    cases = RunDCSampler(10, par);
    data, Flag_c = GoCompetition.case_to_data(cases; t = GoCompetition.ActiveSet)
    size_in = size(data[1][1], 1)
    size_out = GoCompetition.totalconstraints(full_pm.model)
    m = GoCompetition.NeuralNetConstructor(size_in, size_out; hidden_sizes = [10, 10], act = relu, fact=sigmoid)


    opt = ADAM(0.01)
    loss(x, y) = sum(Flux.binarycrossentropy.(m(x), GoCompetition.binding2classifier(y)))

    # Initial loss
    l0 = loss(data[1][1], data[1][2]).data
    Flux.@epochs 5 Flux.train!(loss, Flux.params(m), data, opt)
    lf = loss(data[1][1], data[1][2]).data
    @test lf < l0

    predict(m, x) = convert.(Bool, round.(m(x).data))
    @test predict(m, data[1][1]) isa BitArray

end

@testset "End-to-End Metaoptimization" begin

    trainoptions = Dict(:nepochs => 7, :hidden => [10,10], :opt => ADAM(0.01))
    metaoptions = Optim.Options(iterations=4,  show_trace=true)
    metaoptimizer = ParticleSwarm(n_particles = 3)
    metaobjective = GoCompetition.metaloss
    solver = VerboseSolver()
    network = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")
    genoptions = Dict{String, Any}("dev_load_pd" => 0.1, "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1)
    data_train, data_test = GoCompetition.generate_grid_data(
      network,
      genoptions,
      15,
      GoCompetition.ActiveSet
    );

    res = GoCompetition.run(
      network,
      data_train,
      data_test;
      trainoptions = trainoptions,
      metaoptions = metaoptions,
      metaoptimizer = metaoptimizer,
      metaobjective = metaobjective,
      solver = solver
    )
    @test isa(res[:metaoptres], Optim.MultivariateOptimizationResults)
    @test res[:θ][1] != res[:θ][5] # Check that they have updated

    epochsummary = GoCompetition.epochsummarise(network, res[:trained_model], data_test, res[:θ], res[:train_metrics])
    @test sort(names(epochsummary)) == sort([
      :Iterations_Full,
      :Iterations_Reduced,
      :Time_Reduced,
      :Time_Full,
      :s_time,
      :testing,
      :training,
    ])

end

@testset "Metaloss" begin

    # Test metaloss network is updated
    network = PowerModels.parse_file("./data/case9.m");
    setting = Dict("output" => Dict("branch_flows" => true))
    pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)

    genoptions = Dict{String, Any}("dev_load_pd" => 0.1,
            "dev_gen_max" => 0.9, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);
    data_train, data_test = GoCompetition.generate_grid_data(
      network,
      genoptions,
      15,
      GoCompetition.ActiveSet
    );

    trainoptions = Dict(:nepochs => 7, :hidden => [10,10], :opt => ADAM(0.01))
    size_in = size(data_train[1][1], 1)
    size_out = GoCompetition.totalconstraints(pm_model.model)
    m = GoCompetition.NeuralNetConstructor(size_in, size_out; hidden_sizes = [10, 10], fact = sigmoid)
    Flux.testmode!(m)
    w = GoCompetition.get_w(m)
    l0 = GoCompetition.metaloss(
        w,
        m,
        network,
        data_train,
        identity,
        GoCompetition.ActiveSet
    )

    # If network updated, setting this before hand shouldn't change anything
    GoCompetition.set_x!(network, data_train[1][1])
    l1 = GoCompetition.metaloss(
        w,
        m,
        network,
        data_train,
        identity,
        GoCompetition.ActiveSet
    )

    @test l0 == l1
    p0 = m(data_train[1][1])
    @test all(map(1:50) do (i) p0 == m(data_train[1][1]) end)
    Flux.testmode!(m, false)
    @test any(map(1:50) do (i) p0 != m(data_train[1][1]) end)

    @testset "Classifier and Binding Dict interaction" begin
    # Interaction between binding dict and classifier
      c0 = GoCompetition.predict(m, data_train[1][1], GoCompetition.ActiveSet)
      binding_template = GoCompetition.create_bindingdict_template(data_train[3][2]) # Doesn't matter which one
      b0 = deepcopy(binding_template)
      GoCompetition.classifier2binding!(b0, c0)

      # Now, only the
      ctype = GoCompetition.DC_REL_CONSTRAINT_TYPES[1]
      N1 = filter(x -> x[:ct] == ctype, GoCompetition.ntconstraints(pm_model.model))[1][:N]
      @test b0[ctype...] == c0[1:N1]

      ctype = GoCompetition.DC_REL_CONSTRAINT_TYPES[2]
      N2 = filter(x -> x[:ct] == ctype, GoCompetition.ntconstraints(pm_model.model))[1][:N]
      @test b0[ctype...] == c0[N1+1:N1+N2]

      ctype = GoCompetition.DC_REL_CONSTRAINT_TYPES[3]
      N3 = filter(x -> x[:ct] == ctype, GoCompetition.ntconstraints(pm_model.model))[1][:N]
      @test b0[ctype...] == c0[N1+N2+1:N1+N2+N3]

      ctype = GoCompetition.DC_REL_CONSTRAINT_TYPES[4]
      N4 = filter(x -> x[:ct] == ctype, GoCompetition.ntconstraints(pm_model.model))[1][:N]
      @test b0[ctype...] == c0[N1+N2+N3+1:N1+N2+N3+N4]

      @test length(GoCompetition.DC_REL_CONSTRAINT_TYPES) == 4

      # Check equality constraints remaining are the same
      ctype = (GenericAffExpr{Float64,VariableRef}, MathOptInterface.EqualTo{Float64})
      @test b0[ctype] == binding_template[ctype]
      # Actually, always expect these to be true
      all(b0[ctype] .== true)

    end

end
