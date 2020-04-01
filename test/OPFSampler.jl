@testset "Grid clean-up tests" begin
    base_model = PowerModels.parse_file("./data/pglib_opf_case30_ieee.m")
    pm_model = deepcopy(base_model)
    pm_model["gen"]["1"]["gen_status"] = 0
    pm_model["branch"]["29"]["br_status"] = 0
    GoCompetition.grid_dcopf_cleanup!(pm_model)
    # generator "6" is one of the generators that should be cleand up in this testcase.
    @test haskey(base_model["gen"], "6") != haskey(pm_model["gen"], "6")
    @test haskey(base_model["gen"], "1") != haskey(pm_model["gen"], "1")
    @test haskey(base_model["branch"], "29") != haskey(pm_model["branch"], "29")
    pm_model = deepcopy(base_model)
    pm_model["gen"]["1"]["gen_status"] = 0
    pm_model["branch"]["29"]["br_status"] = 0
    pm_model["gen"]["2"]["pmax"] = pm_model["gen"]["2"]["pmin"] = 0
    pm_model["gen"]["5"]["pmax"] = pm_model["gen"]["5"]["pmin"] = 0
    pm_model["gen"]["5"]["qmax"] = pm_model["gen"]["5"]["qmin"] = 0
    GoCompetition.grid_acopf_cleanup!(pm_model)
    @test haskey(base_model["gen"], "1") != haskey(pm_model["gen"], "1")
    @test haskey(base_model["gen"], "5") != haskey(pm_model["gen"], "5")
    @test haskey(base_model["gen"], "2") == haskey(pm_model["gen"], "2")
    @test haskey(base_model["branch"], "29") != haskey(pm_model["branch"], "29")
end

@testset "OPFSampler.jl tests" begin
    base_model = PowerModels.parse_file("./data/case9.m")
    @testset "PowerModels update functions tests" begin
        pm_model = deepcopy(base_model)
        base_pload = [pm_model["load"][k]["pd"] for k in keys(sort(pm_model["load"]))]
        GoCompetition.set_load_pd!(pm_model, 2 * base_pload)
        new_pload = [pm_model["load"][k]["pd"] for k in keys(sort(pm_model["load"]))]
        @test new_pload == 2 * base_pload
        base_qload = [pm_model["load"][k]["qd"] for k in keys(sort(pm_model["load"]))]
        GoCompetition.set_load_qd!(pm_model, 2 * base_qload)
        new_qload = [pm_model["load"][k]["qd"] for k in keys(sort(pm_model["load"]))]
        @test new_qload == 2 * base_qload
        base_pgmax = [pm_model["gen"][k]["pmax"] for k in keys(sort(pm_model["gen"]))]
        GoCompetition.set_gen_pmax!(pm_model, 2 * base_pgmax)
        new_pgmax = [pm_model["gen"][k]["pmax"] for k in keys(sort(pm_model["gen"]))]
        @test new_pgmax == 2 * base_pgmax
        base_qgmax = [pm_model["gen"][k]["qmax"] for k in keys(sort(pm_model["gen"]))]
        GoCompetition.set_gen_qmax!(pm_model, 2 * base_qgmax)
        new_qgmax = [pm_model["gen"][k]["qmax"] for k in keys(sort(pm_model["gen"]))]
        @test new_qgmax == 2 * base_qgmax
        base_br_x = [pm_model["branch"][k]["br_x"] for k in keys(sort(pm_model["branch"]))]
        base_rate_a = [pm_model["branch"][k]["rate_a"] for k in keys(sort(pm_model["branch"]))]
        GoCompetition.set_dc_branch_param!(pm_model, br_x = 2 * base_br_x,
        rate_a = 2 * base_rate_a)
        new_br_x = [pm_model["branch"][k]["br_x"] for k in keys(sort(pm_model["branch"]))]
        new_rate_a = [pm_model["branch"][k]["rate_a"] for k in keys(sort(pm_model["branch"]))]
        @test new_br_x == 2 * base_br_x
        @test new_rate_a == 2 * base_rate_a
        base_br_x = [pm_model["branch"][k]["br_x"] for k in keys(sort(pm_model["branch"]))]
        base_br_r = [pm_model["branch"][k]["br_r"] for k in keys(sort(pm_model["branch"]))]
        base_rate_a = [pm_model["branch"][k]["rate_a"] for k in keys(sort(pm_model["branch"]))]
        GoCompetition.set_ac_branch_param!(pm_model, br_x = 2 * base_br_x,
        br_r = 2 * base_br_r, rate_a = 2 * base_rate_a)
        new_br_x = [pm_model["branch"][k]["br_x"] for k in keys(sort(pm_model["branch"]))]
        new_br_r = [pm_model["branch"][k]["br_r"] for k in keys(sort(pm_model["branch"]))]
        new_rate_a = [pm_model["branch"][k]["rate_a"] for k in keys(sort(pm_model["branch"]))]
        @test new_br_x == 2 * base_br_x
        @test new_br_r == 2 * base_br_r
        @test new_rate_a == 2 * base_rate_a
    end

    @testset "Grid sample generation tests" begin
        @testset "DC-OPF sample generation:" begin
            pm_model = deepcopy(base_model)
            par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
            "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);
            samples, model = GoCompetition.DC_OPF_sampling(4, par)
            @test isa(model, Dict)
            @test length(samples) == 4
            id1 = 1; id2 = 3;  # picking two of the 4 samples for some tests
            key_list = ["pg_max", "sample_id", "rate_a", "br_x", "price_insensitive_load"]
            #chekcing if it samples have all the expected keys
            @test all([in(kk, key_list) for kk in collect(keys(samples[id2]))])
            @test length(key_list) == length(collect(keys(samples[id2])))
            # checking if samples are different from each other
            @test all(samples[id1]["pg_max"] .!= samples[id2]["pg_max"])
            @test all(samples[id1]["rate_a"] .!= samples[id2]["rate_a"])
            @test all(samples[id1]["br_x"] .!= samples[id2]["br_x"])
            @test all(samples[id1]["price_insensitive_load"] .!=
            samples[id2]["price_insensitive_load"])
            @testset "dictionary order tests" begin
                base_pd = [base_model["load"][k]["pd"] for k in keys(sort(base_model["load"]))]
                @test all((1-par["dev_load_pd"]) .* base_pd .<= samples[1]["price_insensitive_load"] .<= (1+par["dev_load_pd"]) .* base_pd)
                base_pgmax = [base_model["gen"][k]["pmax"] for k in keys(sort(base_model["gen"]))]
                @test all((1-par["dev_gen_max"]) .* base_pgmax .<= samples[1]["pg_max"] .<= (1+par["dev_gen_max"]) .* base_pgmax)
                base_rate_a = [base_model["branch"][k]["rate_a"] for k in keys(sort(base_model["branch"]))]
                @test all((1-par["dev_rate_a"]) .* base_rate_a .<= samples[1]["rate_a"] .<= (1+par["dev_rate_a"]) .* base_rate_a)
                base_br_x = [base_model["branch"][k]["br_x"] for k in keys(sort(base_model["branch"]))]
                @test all((1-par["dev_br_x"]) .* base_br_x .<= samples[1]["br_x"] .<= (1+par["dev_br_x"]) .* base_br_x)
            end
        end
        @testset "AC-OPF sample generation:" begin
            pm_model = deepcopy(base_model)
            par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
            "dev_load_qd" => 0.1, "dev_pgen_max" => 0.1, "dev_qgen_max" => 0.1,
            "dev_rate_a" => 0.1, "dev_br_x" => 0.1, "dev_br_r" => 0.1);
            samples, model = GoCompetition.AC_OPF_sampling(4, par)
            @test isa(model, Dict)
            @test length(samples) == 4
            id1 = 1; id2 = 3;  # picking two of the 4 samples for some tests
            key_list = ["pg_max", "qg_max", "sample_id", "rate_a", "br_x", "br_r",
             "price_insensitive_pload", "price_insensitive_qload"]
            #chekcing if it samples have all the expected keys
            @test all([in(kk, key_list) for kk in collect(keys(samples[id2]))])
            @test length(key_list) == length(collect(keys(samples[id2])))
            # checking if samples are different from each other
            @test all(samples[id1]["pg_max"] .!= samples[id2]["pg_max"])
            @test all(samples[id1]["qg_max"] .!= samples[id2]["qg_max"])
            @test all(samples[id1]["rate_a"] .!= samples[id2]["rate_a"])
            @test all(samples[id1]["br_x"] .!= samples[id2]["br_x"])
            @test sum(samples[id1]["br_r"] .!= samples[id2]["br_r"]) > 0
            @test all(samples[id1]["price_insensitive_pload"] .!=
            samples[id2]["price_insensitive_pload"])
            @test all(samples[id1]["price_insensitive_qload"] .!=
            samples[id2]["price_insensitive_qload"])
            @testset "dictionary order tests" begin
                base_pd = [base_model["load"][k]["pd"] for k in keys(sort(base_model["load"]))]
                @test all((1-par["dev_load_pd"]) .* base_pd .<= samples[1]["price_insensitive_pload"] .<= (1+par["dev_load_pd"]) .* base_pd)
                base_qd = [base_model["load"][k]["qd"] for k in keys(sort(base_model["load"]))]
                @test all((1-par["dev_load_qd"]) .* base_qd .<= samples[1]["price_insensitive_qload"] .<= (1+par["dev_load_qd"]) .* base_qd)
                base_pgmax = [base_model["gen"][k]["pmax"] for k in keys(sort(base_model["gen"]))]
                @test all((1-par["dev_pgen_max"]) .* base_pgmax .<= samples[1]["pg_max"] .<= (1+par["dev_pgen_max"]) .* base_pgmax)
                base_qgmax = [base_model["gen"][k]["qmax"] for k in keys(sort(base_model["gen"]))]
                @test all((1-par["dev_qgen_max"]) .* base_qgmax .<= samples[1]["qg_max"] .<= (1+par["dev_qgen_max"]) .* base_qgmax)
                base_rate_a = [base_model["branch"][k]["rate_a"] for k in keys(sort(base_model["branch"]))]
                @test all((1-par["dev_rate_a"]) .* base_rate_a .<= samples[1]["rate_a"] .<= (1+par["dev_rate_a"]) .* base_rate_a)
                base_br_x = [base_model["branch"][k]["br_x"] for k in keys(sort(base_model["branch"]))]
                @test all((1-par["dev_br_x"]) .* base_br_x .<= samples[1]["br_x"] .<= (1+par["dev_br_x"]) .* base_br_x)
                base_br_r = [base_model["branch"][k]["br_r"] for k in keys(sort(base_model["branch"]))]
                @test all((1-par["dev_br_r"]) .* base_br_r .<= samples[1]["br_r"] .<= (1+par["dev_br_r"]) .* base_br_r)
            end
        end
    end

    @testset "Running sampler tests" begin
        @testset "Running DC-Sampler:" begin
            pm_model = deepcopy(base_model)
            par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
            "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);
            samples = RunDCSampler(4, par)
            @test all([haskey(samples[k], "OPF_output") for k in 1:4])
            if (samples[1]["OPF_output"]["termination_status"] == LOCALLY_SOLVED) &
                (samples[3]["OPF_output"]["termination_status"] == LOCALLY_SOLVED)

                @test samples[1]["OPF_output"]["objective"] !=
                samples[3]["OPF_output"]["objective"]
            elseif (samples[1]["OPF_output"]["termination_status"] == OPTIMAL) &
                (samples[3]["OPF_output"]["termination_status"] == OPTIMAL)

                @test samples[1]["OPF_output"]["objective"] !=
                samples[3]["OPF_output"]["objective"]
            end
        end
        @testset "Running AC-Sampler:" begin
            pm_model = deepcopy(base_model)
            par = Dict("case_network" => pm_model, "dev_load_pd" => 0.1,
            "dev_load_qd" => 0.1, "dev_pgen_max" => 0.1, "dev_qgen_max" => 0.1,
            "dev_rate_a" => 0.1, "dev_br_x" => 0.1, "dev_br_r" => 0.1);
            samples = RunACSampler(4, par)
            @test all([haskey(samples[k], "OPF_output") for k in 1:4])
            if (samples[1]["OPF_output"]["termination_status"] == LOCALLY_SOLVED) &
                (samples[3]["OPF_output"]["termination_status"] == LOCALLY_SOLVED)

                @test samples[1]["OPF_output"]["objective"] !=
                samples[3]["OPF_output"]["objective"]
            elseif (samples[1]["OPF_output"]["termination_status"] == OPTIMAL) &
                (samples[3]["OPF_output"]["termination_status"] == OPTIMAL)

                @test samples[1]["OPF_output"]["objective"] !=
                samples[3]["OPF_output"]["objective"]
            end
        end
    end
end
