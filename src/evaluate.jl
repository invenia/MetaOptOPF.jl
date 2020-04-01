# Takes a given PowerModels model and a trained Flux model along with a series of
# D test data examples. Outputs three timings, standard, exact, and model.
# Each timing output is an Array{..., 2} of size N, D
function evaluate(N::Int, pm_model, model, data)
        solve_time_std = zeros(length(data), N)
        solve_time_exact = zeros(length(data), N)
        solve_time_model = zeros(length(data), N)
        for j = 1:N
            for i = 1:length(data)
                network_data_ = deepcopy(pm_model)
                # Perturb the network data into a distinct example.
                set_x!(network_data_, data[i][1])
                # Standard solve using whatever initialization is default
                opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
                #solve_time_std[i, j] = opf_result["solve_time"]
                solve_time_std[i, j] = Niter
                # Solve, using the initialization as the solution.
                set_y!(network_data_, data[i][2])
                opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
                #solve_time_exact[i, j] = opf_result["solve_time"]
                solve_time_exact[i, j] = Niter
                # Solve, usin the output of the Flux model as the initialization.
                set_y!(network_data_, model(data[i][1]).data)
                opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
                #solve_time_model[i, j] = opf_result["solve_time"]
                solve_time_model[i, j] = Niter
            end
        end
        return solve_time_std, solve_time_exact, solve_time_model
end

# TODO: Refactor these into a single function call with the above.
function evaluate(N::Int, pm_model, model, metamodel, data)
    solve_time_std = zeros(length(data), N)
    solve_time_exact = zeros(length(data), N)
    solve_time_model = zeros(length(data), N)
    solve_time_metamodel = zeros(length(data), N)
    for j = 1:N
        for i = 1:length(data)
            network_data_ = deepcopy(pm_model)
            # Perturb the network data into a distinct example.
            set_x!(network_data_, data[i][1])
            # Standard solve using whatever initialization is default
            opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
            # Solve, using the initialization as the solution.
            set_y!(network_data_, data[i][2])
            opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
            # Solve, usin the output of the Flux model as the initialization.
            set_y!(network_data_, model(data[i][1]).data)
            opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
            set_y!(network_data_, metamodel(data[i][1]).data)
            opf_result, Niter, ctime = VerboseSolver()(s -> run_dc_opf(network_data_, s))
        end
    end
    return solve_time_std, solve_time_exact, solve_time_model, solve_time_metamodel
end

function evaluate(m::Chain, data, t::Type{ActiveSet})
    N_test = length(data)
    ŷ = hcat([predict(m, data[i][1], t) .== binding2classifier(data[i][2]) for i in 1:N_test]...)
    mean(ŷ, dims = 1) # Placeholder for whatever metrics we care about
end

function metaevaluate(m::Chain, network, data, t::Type{ActiveSet}; solver::Solver = VerboseSolver())
    _network = deepcopy(network)
    GoCompetition.set_x!(_network, data[1])
    c0 = GoCompetition.predict(m, data[1], ActiveSet)
    b0 = GoCompetition.create_bindingdict_template(data[2])
    GoCompetition.classifier2binding!(b0, c0, n_gen_lb=length(_network["gen"]))
    out = GoCompetition.objective_extend!(_network, b0)
    nt, ct, st = out[:niter], out[:ctime], out[:stime]
    _network = deepcopy(network)
    GoCompetition.set_x!(_network, data[1])
    opf_result, n0, ctime = solver(s -> run_dc_opf(_network, s))
    return DataFrame(:N_0 => n0, :t_0 => ctime, :N_trained => nt, :c_meta => ct, :s_time => st)
end

# epoch summarise takes weights found during training
# For each obs in data (i.e. data_train), it gets the metaevaluation metrics
# These are ctime, n_epochs, etc.
# It also compares with the full (standard OPF solve) approach
# For all observations it sums then renames
function epochsummarise(network, m_template::Chain, data, θ, l::DataFrame)

    setting = Dict("output" => Dict("branch_flows" => true))
    pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)

    performance = map(θ) do w
        GoCompetition.set_w!(m_template, w)
        epochmeta = vcat(map(data) do (dt) metaevaluate(m_template, network, dt, ActiveSet) end ...)
        Dict(colname => sum(epochmeta[colname]) for colname in names(epochmeta))
    end
    d = hcat(vcat(DataFrame.(performance)...), l)
    rename!(d,
        :N_0 => :Iterations_Full,
        :N_trained => :Iterations_Reduced,
        :c_meta => :Time_Reduced,
        :t_0 => :Time_Full
    )
    return d
end



# Outputs summary statistics compariing the performance of the three solutions.
function summarise(solve_time_std, solve_time_exact, solve_time_model)
    n = size(solve_time_std, 1)
    exact_vs_std = ( sum(solve_time_std, dims=1) .- sum(solve_time_exact, dims=1) ) ./ sum(solve_time_std, dims=1)
    model_vs_std = ( sum(solve_time_std, dims=1) .- sum(solve_time_model, dims=1) ) ./ sum(solve_time_std, dims=1)
    μ1 = mean(exact_vs_std);
    σ1 = std(exact_vs_std);
    CI1 = 1.96 * σ1 ./ sqrt(n)
    μ2 = mean(model_vs_std);
    σ2 = std(model_vs_std);
    CI2 = 1.96 * σ2 ./ sqrt(n)
    @printf("%s%.f%s%.2f%s\n", "Gain using exact solution: ", μ1 * 100.0, " +/- ", CI1 * 100, "%")
    @printf("%s%.f%s%.2f%s\n", "Gain using Model 1 solution: ", μ2 * 100.0, " +/- ", CI2 * 100, "%")
    return (μ1, σ1), (μ2, σ2)
end

function summarise(solve_time_std, solve_time_exact, solve_time_model, solve_time_metamodel)
    n = size(solve_time_std, 1)
    exact_vs_std = ( sum(solve_time_std, dims=1) .- sum(solve_time_exact, dims=1) ) ./ sum(solve_time_std, dims=1)
    model_vs_std = ( sum(solve_time_std, dims=1) .- sum(solve_time_model, dims=1) ) ./ sum(solve_time_std, dims=1)
    metamodel_vs_std = ( sum(solve_time_std, dims=1) .- sum(solve_time_metamodel, dims=1) ) ./ sum(solve_time_std, dims=1)
    μ1 = mean(exact_vs_std);
    σ1 = std(exact_vs_std);
    CI1 = 1.96 * σ1 ./ sqrt(n);
    μ2 = mean(model_vs_std);
    σ2 = std(model_vs_std);
    CI2 = 1.96 * σ2 ./ sqrt(n)
    μ3 = mean(metamodel_vs_std);
    σ3 = std(metamodel_vs_std);
    CI3 = 1.96 * σ3 ./ sqrt(n)



    @printf("%s%.f%s%.2f%s\n", "Gain using exact solution: ", μ1 * 100.0, " +/- ", CI1 * 100, "%")
    @printf("%s%.f%s%.2f%s\n", "Gain using Model 1 solution: ", μ2 * 100.0, " +/- ", CI2 * 100, "%")
    @printf("%s%.f%s%.2f%s\n", "Gain using Model 2 solution: ", μ3 * 100.0, " +/- ", CI3 * 100, "%")
    return (μ1, σ1), (μ2, σ2), (μ3, σ3)
end
