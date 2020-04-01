using Distributed

using GoCompetition
using Ipopt
using JuMP
using PowerModels
using Flux
using Pkg

@testset "Metaloss [Distributed]" begin

    # Test metaloss network is updated
    network = PowerModels.parse_file("./data/case9.m");
    setting = Dict("output" => Dict("branch_flows" => true))
    pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    genoptions = Dict{String, Any}("dev_load_pd" => 0.1,
            "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);
    data_train, data_test = GoCompetition.generate_grid_data(
      network,
      genoptions,
      15,
      GoCompetition.ActiveSet
    );
    size_in = size(data_train[1][1], 1)
    size_out = GoCompetition.totalconstraints(pm_model.model)
    m = GoCompetition.NeuralNetConstructor(size_in, size_out; hidden_sizes = [10, 10], fact = sigmoid)
    Flux.testmode!(m)

    rmprocs(workers())
    @test nprocs() == 1 # Assertion for the test, as this should be 1
    transform = identity
    pool = CachingPool(workers())
    res1 = GoCompetition.metaloss(
        GoCompetition.get_w(m),
        m,
        network,
        data_test,
        transform,
        GoCompetition.ActiveSet;
        solver = GoCompetition.VerboseSolver(),
        obj_quantity=:niter,
        pool = pool
    )

    addprocs(2)
    # These imports need to stay here to add them to the spawned processes
    @everywhere begin
      using Pkg
      Pkg.activate("../")  # required
    end

    @everywhere begin
      using GoCompetition
      using Ipopt
      using JuMP
      using PowerModels
    end

    pool = CachingPool(workers())
    res2 = GoCompetition.metaloss(
    GoCompetition.get_w(m),
        m,
        network,
        data_test,
        transform,
        GoCompetition.ActiveSet;
        solver = GoCompetition.VerboseSolver(),
        obj_quantity=:niter,
    pool = pool)

    @test res1 == res2


end
