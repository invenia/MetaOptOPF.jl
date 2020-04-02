using Distributed
using ArgParse
using Flux
using MLDataUtils
using Statistics
using Optim
using OrderedCollections # Define sort on dict
using DataFrames
using ArgParse
using JLD2
using MathOptInterface
using Flux.Tracker: gradient, update!
using Dates
using Plots
using BSON


@info "Parsing args..."
s = ArgParseSettings()
@add_arg_table s begin
    "--casepath"
        help = "Path to the case"
        required = true
    "--datapath"
        help = "Path to the data"
        required = true
    "--P"
        help = "number of processes to spawn. Typically choose the number of vCPUS" # NOte that this can be done by julia -p X
        arg_type = Int
        required = true
    "--N"
        help = "Number of experiments to run."
        arg_type = Int
        default = 10
    "--min_epochs"
        help = "Minimum number of epochs to train"
        arg_type = Int
        default = 50
    "--lr"
        help = "Learning rate for ADAM optimizer ('nu')"
        arg_type = Float64
        default = 1e-4
    "--batchsize"
        help = "The batch size"
        arg_type = Int
        required = true
    "--metatrainsize"
        help = "The metatraining size"
        arg_type = Int
        required = true
    "--splits"
        help = "
          The percentage of data to divide between datasets.
          e.g. training and validation (and remaining into testing) e.g. (0.8, 0.1)
          "
        arg_type = Tuple{Float64, Float64}
        default = (0.7, 0.2)
    "--PSOparticles"
        help = "The number of particles to use in PSO"
        arg_type = Int
        required = true
    "--PSOiterations"
        help = "The number of particles to use in PSO"
        arg_type = Int
        required = true
    "--patience"
        help = "Number of iterations after min_epochs to allow for increasing loss. "
        arg_type = Int
        default = 10
    "--penalty_factor"
        help = "Penalty factor multiplying the maximum number of average active constrains"
        arg_type = Float64
        default = 2.0
end

parsedargs = parse_args(ARGS, s)
println("Parsed args:")
for pa in parsedargs
    println("  $(pa[1])  =>  $(pa[2])")
end

# Compute parameters
P = parsedargs["P"]

# Data Parameters
casepath = parsedargs["casepath"]
datapath = parsedargs["datapath"]

SPLITS = parsedargs["splits"]
METATRAINSIZE = parsedargs["metatrainsize"]

# Experiment parameters
N = parsedargs["N"]

# Classical learning parameters
MIN_EPOCHS = parsedargs["min_epochs"]
BATCHSIZE = parsedargs["batchsize"]
PATIENCE = parsedargs["patience"]
lr = parsedargs["lr"]

# PSO Parameters
PSO_P = parsedargs["PSOparticles"]
PSO_N = parsedargs["PSOiterations"]

# Penalty parameter
PENALTY_FACTOR = parsedargs["penalty_factor"]

# Validation
(!isfile(casepath) || !isfile(datapath)) && error("At least one path in args is not a file")
(BATCHSIZE <= 1) && error("The batch size should be greater than 1")

@info "Spawning workers..."
addprocs(P)
pool = WorkerPool(workers())
@everywhere begin
  using Pkg
  Pkg.activate("../../")  # required
end

@everywhere begin
  using GoCompetition
  using Ipopt
  using JuMP
  using PowerModels
  PowerModels.silence()
end

makebatch(batch::Vector{Array{T, N}}) where {T,N} = cat(batch..., dims = ndims(batch) + 1)
makebatch(x, y) = (makebatch(x), makebatch(GoCompetition.binding2classifier.(y)))
@everywhere makebatch(batch::Array{T, N}) where {T,N} = reshape(batch, (size(batch)...,1))

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function callback(x)
    println(" * Iteration:       ", x[end].iteration, "    ", minimum([x[i].value for i in 1:length(x)]))
    flush(stdout)
    return false
end

function flushprintln(x...)
    println(x...)
    flush(stdout)
end

@info "Building network..."
network = PowerModels.parse_file(casepath)
#GoCompetition.grid_acopf_cleanup!(network)

setting = Dict("output" => Dict("branch_flows" => true))
pm_model = build_model(network, ACPPowerModel, PowerModels.post_opf, setting=setting);

@eval @load $datapath
@show(size(data))
x_orig = GoCompetition.get_x(network)
x_orig[x_orig .== 0] .= 1.0
transform = x -> (x - x_orig) ./ x_orig
batchtransform = batch -> mapslices(transform, batch; dims = 1)
opt = ADAM(lr)

function run_before()
    x = false
    function has_run_before()
        if !x
            x = true
            flushprintln("Setting flag")
            return false
        else
            return true
        end
    end
    return has_run_before
end

function flushprintln(d::Dict)
    for (key, value) in d
        flushprintln(key, " ==> ", value)
    end
end

precompilationcheck = run_before()

function _main(
    model::Chain,
    data::Tuple;
    do_stop::Function,
    experiment::Int = 1,
    prefix::String = "test",
    skip_meta::Bool = true,
    )

    data_train, data_validation, data_test = data

    # Model
    ps = Flux.params(model)

    # Loss functions
    _bloss(bX, bY) = mean(Statistics.mean(Flux.binarycrossentropy.(model(batchtransform(bX)), bY, ϵ = eps(Float32)); dims = 1))
    _loss(x, y) = Statistics.mean(Flux.binarycrossentropy.(model(transform(x)), y))

    n_pot_active_constr_train = count(i -> (i>0), mapreduce(d -> GoCompetition.binding2classifier(d[2]), +, data_train))
    max_active_constr_train = maximum(map(d -> sum(GoCompetition.binding2classifier(d[2])), data_train))
    thr = max_active_constr_train * PENALTY_FACTOR

    flushprintln("$prefix: n_pot_active_constr_train = ", n_pot_active_constr_train)
    flushprintln("$prefix: max_active_constr_train = ", max_active_constr_train)

    function _metaloss(w)
        data_meta_train = shuffleobs(data_train)[1:METATRAINSIZE]
        mdl = deepcopy(model)
        GoCompetition.set_w!(mdl, w)
        if mean(map(d -> sum(GoCompetition.predict(mdl, makebatch(transform(d[1])), GoCompetition.ActiveSet)), data_meta_train)) > thr
            return Inf
        else
            return GoCompetition.metaloss(w, model, network, data_meta_train, x -> makebatch(transform(x)), GoCompetition.ActiveSet; solver = GoCompetition.VerboseSolver(), obj_quantity=:ctime, pool = pool)
        end
    end

    valX, valY = unzip(data_validation)
    epochs = 0
    l_val = NaN
    while !do_stop(epochs, l_val)
        trainX, trainY = unzip(shuffleobs(data_train))
        for (batch_X, batch_Y) in eachbatch((trainX, trainY), size = BATCHSIZE)
            _bX, _bY = makebatch(batch_X, batch_Y)
            gs = gradient(ps) do
                _bloss(_bX, _bY)
            end
            update!(opt, ps, gs)
        end
        epochs += 1
        Flux.testmode!(model)
        l_train = mean(map(b -> _bloss(makebatch(b...)...), eachbatch((trainX, trainY), size = BATCHSIZE)))
        l_val = mean(map(b -> _bloss(makebatch(b...)...), eachbatch((valX, valY), size = BATCHSIZE)))
        Flux.testmode!(model, false)
        flushprintln("loss_train = ", l_train, "\t", "loss_val = ", l_val)
    end

    flushprintln("Stopped after patience exhausted in $epochs epochs [Patience started at $PATIENCE]")

    testX, testY = unzip(data_test)
    Flux.testmode!(model)
    # Save model
    model_pre = deepcopy(model)




    loss_test = mean(map(b -> _bloss(makebatch(b...)...), eachbatch((testX, testY), size = 1)))
    metaloss_test_pre = GoCompetition.metaloss(GoCompetition.get_w(model), model, network, data_test, x -> makebatch(transform(x)), GoCompetition.ActiveSet; solver = GoCompetition.VerboseSolver(), obj_quantity=:ctime, pool = pool)
    mean_n_active_cnstr_test_pre = mean(map(d -> sum(GoCompetition.predict(model, makebatch(transform(d[1])), GoCompetition.ActiveSet)), data_test))
    metaloss_test_post = NaN
    mean_n_active_cnstr_test_post = NaN

    flushprintln("$prefix: loss_test = ", loss_test)
    flushprintln("$prefix: metaloss_test (pre) = ", metaloss_test_pre)
    flushprintln("$prefix: mean_n_active_cnstr_test (pre) = ", mean_n_active_cnstr_test_pre)

    if !skip_meta
      res = optimize(_metaloss, GoCompetition.get_w(model), ParticleSwarm(n_particles = PSO_P, all_from_init=true, delta=abs.(GoCompetition.get_w(model))), Optim.Options(iterations=PSO_N, show_trace=false, store_trace = true, extended_trace = true, callback = callback))
      model_post = deepcopy(model)
      GoCompetition.set_w!(model_post, res.trace[end].metadata["x"])
      metaloss_test_post = GoCompetition.metaloss(res.trace[end].metadata["x"], model, network, data_test, x -> makebatch(transform(x)), GoCompetition.ActiveSet; solver = GoCompetition.VerboseSolver(), obj_quantity=:ctime, pool = pool)
      mean_n_active_cnstr_test_post = mean(map(d -> sum(GoCompetition.predict(model_post, makebatch(transform(d[1])), GoCompetition.ActiveSet)), data_test))

      flushprintln("$prefix: metaloss_test (post) = ", metaloss_test_post)
      flushprintln("$prefix: mean_n_active_cnstr_test (post) = ", mean_n_active_cnstr_test_post)
    end

    results = Dict(
      "$prefix"*"loss_test" => loss_test.data,
      "$prefix"*"metaloss_test_pre" => metaloss_test_pre,
      "$prefix"*"mean_n_active_cnstr_test_pre" => mean_n_active_cnstr_test_pre,
      "$prefix"*"metaloss_test_post" => metaloss_test_post,
      "$prefix"*"mean_n_active_cnstr_test_post" => mean_n_active_cnstr_test_post
    )

    try


      fdir = "./experiments/experiment_$experiment/$prefix/"
      mkpath(fdir)


      JLD2.@save "./$fdir"*"results.jld2" results

      weights_pre = Tracker.data.(params(model_pre));
      BSON.@save "./$fdir"*"model_pre.bson" weights_pre
      if !skip_meta
          model_post = deepcopy(model)
          GoCompetition.set_w!(model_post, res.trace[end].metadata["x"])
          weights_post = Tracker.data.(params(model_post));
          BSON.@save "./$fdir"*"model_post.bson" weights_post
      end
      JLD2.@save "./$fdir"*"data_test.jld2" data_test # potentially can save index

      fig1 = heatmap(hcat(map(data_test) do (dt) GoCompetition.binding2classifier(dt[2]) end ...), xlabel = "Observation", ylabel = "Constraint ID")
      fig2 = heatmap(hcat(map(data_test) do (dt) GoCompetition.predict(model_pre, makebatch(transform(dt[1])), GoCompetition.ActiveSet) end ...), xlabel = "Observation", ylabel = "Constraint ID")

      savefig(fig1, fdir * "fig1.png")
      savefig(fig2, fdir * "fig2.png")

      if !skip_meta
        fig3 = heatmap(hcat(map(data_test) do (dt) GoCompetition.predict(model_post, makebatch(transform(dt[1])), GoCompetition.ActiveSet) end ...), xlabel = "Observation", ylabel = "Constraint ID")
        savefig(fig3, fdir * "fig3.png")
      end

    catch
      @warn("Saving figures went wrong")
    end

    return results

end


@info "Starting main loop..."
for n = 1:N
    # try
        flushprintln("===START OF EXPERIMENT===")
        @info(Dates.now())
        flushprintln("===============")
        _data = shuffleobs(data) # Scoping errors with views
        _data = map(d -> Pair(d[1], d[2]), _data) # Parse data into a Pair of x and Y values.

        data_train, data_validation, data_test = splitobs(_data, at = SPLITS)

        @info(size(data_train)) # Train
        @info(size(data_validation)) # Validation
        @info(size(data_test)) # Test

        size_in = size(data_train[1][1], 1)
        size_out = GoCompetition.totalconstraints(pm_model.model)




        model1 = Chain(Dense(size_in, 50), BatchNorm(50, relu), Dropout(0.4),
                    Dense(50, 50), BatchNorm(50, relu), Dropout(0.4),
                    Dense(50, size_out, σ))

        model2 = Chain(Dense(size_in, 50, relu), Dropout(0.4),
                    Dense(50, 50, relu), Dropout(0.4),
                    Dense(50, size_out, σ))

        # Run the function once first to get all precompilation done
        flushprintln("===FULL===")
        !precompilationcheck() && GoCompetition.metaloss(GoCompetition.get_w(model2), model2, network, data_test, x -> makebatch(transform(x)), GoCompetition.ActiveSet; solver = GoCompetition.VerboseSolver(), obj_quantity=:ctime, pool = pool)

        metaloss_test_full = GoCompetition.metaloss(GoCompetition.get_w(model2), model2, network, data_test, x -> makebatch(transform(x)), GoCompetition.ActiveSet; solver = GoCompetition.VerboseSolver(), obj_quantity=:ctime, pool = pool, nn_threshold=0.0)
        mean_n_active_cnstr_test_full = mean(map(d -> sum(GoCompetition.binding2classifier(d[2])), data_test))
        flushprintln("metaloss_test (full) = ", metaloss_test_full)
        flushprintln("mean_n_active_cnstr_test (full) = ", mean_n_active_cnstr_test_full)

        prefix = "full"
        fdir = "./experiments/experiment_$n/$prefix/"
        mkpath(fdir)
        results0 = Dict(
          "$prefix"*"metaloss_test" => metaloss_test_full,
          "$prefix"*"mean_n_active_cnstr_test" => mean_n_active_cnstr_test_full,
        )
        JLD2.@save "./$fdir"*"$prefix"*"results.jld2" results0

        # Stop function model 1:
        should_stop! = GoCompetition.stopping(Float32, patience = PATIENCE)

        flushprintln("===MODEL1===")
        results1 = _main(
            model1,
            (data_train, data_validation, data_test);
            do_stop = (epochs, l_val) -> epochs < MIN_EPOCHS ? false : should_stop!(l_val.data),
            experiment = n,
            prefix = "model1",
            skip_meta = false,
        )

        flushprintln("===MODEL2===")
        results2 = _main(
            model2,
            (data_train, data_validation, data_test);
            do_stop = (epochs, l_val) -> epochs < MIN_EPOCHS ? false : true,
            experiment = n,
            prefix = "model2",
            skip_meta = true,
        )

        flushprintln("===RESULTS===")
        flushprintln(results0)
        flushprintln(results1)
        flushprintln(results2)
        flushprintln("===END OF EXPERIMENT===")

    # catch
    #     @warn("Something went wrong in experiment. ")
    # end
end
