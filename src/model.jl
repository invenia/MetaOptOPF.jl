using Flux.Tracker: update!, gradient

get_w(model) = convert(Array{Float64}, vcat(map(x -> vec(x.data), Flux.params(model))...))

""" set_w!(model, w)

Takes a neural network `model` and weights w and sets the parameters of model to w.

"""
function set_w!(model, w)
    sizes = map(x -> size(x), Flux.params(model))
    nelem = map(x -> prod(x), sizes)
    remaining_w = deepcopy(w)
    ps = map(nelem) do n
        layer = reshape(popfirstn!(remaining_w, n), sizes[1])
        popfirstn!(sizes, 1)
        layer
    end
    Flux.loadparams!(model, ps)
end

# Helper function
function popfirstn!(x, k)
    popped = x[1 : k]
    deleteat!(x, 1:k)
    return popped
end

# Helper function to return a neural network based on some default heuristics.
function NNmodel(N_in::Int, N_out::Int)
  a1 = N_in
  a2 = 10
  a3 = 10
  a4 = N_out
  model = Chain(Dense(a1, a2, relu), Dense(a2, a3, relu), Dense(a3, a4))
  return model
end

function NeuralNetConstructor(insize, outsize; hidden_sizes, act=relu, fact=sigmoid)
    prev_size = insize
    layers = []
    for hidden_size in hidden_sizes
        push!(layers, Dense(prev_size, hidden_size, act))
        push!(layers, Dropout(0.3))
        prev_size = hidden_size
    end
    push!(layers, Dense(hidden_sizes[end], outsize, fact))  #final linear layer
    return Chain(layers...)
end



function metaloss(w, model, pm, data; solver::Solver = VerboseSolver())
    mdl = deepcopy(model)
    set_w!(mdl, w)
    N = length(data)
    loss = 0
    network_data_ = deepcopy(pm)
    for i = 1:N
        set_x!(network_data_, data[i][1])
        set_y!(network_data_, mdl(data[i][1]).data)
        opf_result, Niter, ctime = solver(s -> run_dc_opf(network_data_, s))
        loss += Niter
    end
    return loss
end

function metaloss(
        w,
        model::Chain,
        network::Dict,
        data,
        transform::Function,
        t::Type{GoCompetition.ActiveSet};
        solver::Solver = VerboseSolver(),
        obj_quantity = :niter,
        n = 1,
        pool = WorkerPool(workers()),
        nn_threshold = 0.5
        )

    mdl = deepcopy(model)
    set_w!(mdl, w)
    N = length(data)
    b0 = create_bindingdict_template(data[1][2])

    function metapipeline(d)
        _network = deepcopy(network)
        set_x!(_network, d[1])
        _c0 = predict(mdl, transform(d[1]), t, nn_threshold=nn_threshold)
        _b0 = deepcopy(b0)
        classifier2binding!(_b0, _c0, n_gen_lb=length(_network["gen"]))
        obj = objective_extend!(_network, _b0, n=n)
         # Leave this commented for now. Can always add back in or add a debug mode
        #@show obj[obj_quantity]
        obj[obj_quantity]
    end
    l = pmap(metapipeline, pool, data)
    return sum(l)
end



loss(m, x, y::Dict) = Statistics.mean(Flux.binarycrossentropy.(m(x), binding2classifier(y)))

#predict(m::Chain, x, ::Type{ActiveSet}) = convert.(Bool, round.(m(x).data))
predict(m::Chain, x, ::Type{ActiveSet}; nn_threshold=0.5) = m(x).data .>= nn_threshold


"""
    objective(network::Dict, binding_status::Dict; cfunc=extend_binding_status!, threshold=1.0e-5, n=1)

Objective function returning the total number of iterations used to solve a reduced OPF
problem until all original constraints are satisfied. The constraints kept in the formulation
are updated through the update rule expressed by 'cfunc'.
Warm start is currecntly not used (commented out) as it can lead to diverging iterates in Ipopt.
# Arguments
- `network::Dict`: PowerModels network
- `binding_status::Dict`: starting binding status
- `cfunc::function`:: function type for updating binding status
- `threshold::Float64`: Threshold value for the left and right hand side comparison
  of inequality constraints. Default value is 1.0e-5.
- `n`: how many times to repeat OPF calculation from the average quantity is computed. Default is 1.

# Returns
- `(niter = niter, ctime = ctime, stime = stime)`: Total number of optimization steps, total CPU times
returned by Ipopt and total CPU time returned by PowerModels
"""
function objective(
        network::Dict,
        binding_status::Dict;
        cfunc = extend_binding_status!,
        solver::Solver = VerboseSolver(),
        threshold = 1.0e-5,
        n = 1
    )

    n <= 0 && error("n must be greater than 0!")
    setting = Dict("output" => Dict("branch_flows" => true))
    power_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
    remove_nonbinding_constraints!(power_model, binding_status)
    opf_result, ni, ct = solver(s -> optimize_model!(power_model, s))
    st = opf_result["solve_time"]
    for i = 1:n-1
        opf_result_i, ni_i, ct_i = solver(s -> optimize_model!(power_model, s))
        st_i = opf_result_i["solve_time"]
        ni += ni_i
        ct += ct_i
        st += st_i
    end
    niter = ni / n
    ctime = ct / n
    stime = st / n
    siter = 0
    while cfunc(deepcopy(network), opf_result, binding_status, threshold=threshold) > 0
        siter += 1
        # warmstart!(network, opf_result["solution"])
        power_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)
        remove_nonbinding_constraints!(power_model, binding_status)
        opf_result, ni, ct = solver(s -> optimize_model!(power_model, s))
        st = opf_result["solve_time"]
        for i = 1:n-1
            opf_result_i, ni_i, ct_i = solver(s -> optimize_model!(power_model, s))
            st_i = opf_result_i["solve_time"]
            ni += ni_i
            ct += ct_i
            st += st_i
        end
        niter += ni / n
        ctime += ct / n
        stime += st / n
    end
    return (niter = niter, ctime = ctime, stime = stime, siter = siter)
end


objective_extend!(args...; kwargs...) = objective(args...; cfunc = extend_binding_status!, kwargs...)
objective_update!(args...; kwargs...) = objective(args...; cfunc = update_binding_status!, kwargs...)

function solvestats(solveinfo::AbstractString; flush = false)
    Niter = filter(
        line -> occursin(r"^Number of Iterations*", line),
        readlines(open(solveinfo)))[1]
    Niter = parse(Int, split(Niter, ":")[2])

    ctime = filter(
        line -> occursin(r"^Total CPU secs in IPOPT*", line),
        readlines(open(solveinfo)))[1]
    ctime = parse(Float64, split(ctime, "=")[2])
    flush ? flushstats!() : nothing
    return Niter, ctime
end

n_parameters(m, k) = sum(map(m) do (layer) length(getfield(layer, k)) end)

function run(
        network,
        data_train,
        data_test;
        trainoptions = Dict(:nepochs => 5, :hidden => [10,10], :opt => ADAM(0.01)),
        metaobjective = GoCompetition.metaloss,
        metaoptimizer = ParticleSwarm(n_particles = 20),
        metaoptions = Optim.Options(iterations=10,  show_trace=true),
        solver::Solver = VerboseSolver()
    )

    setting = Dict("output" => Dict("branch_flows" => true))
    pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting)

    size_in = size(data_train[1][1], 1)
    size_out = GoCompetition.totalconstraints(pm_model.model)
    m = GoCompetition.NeuralNetConstructor(size_in, size_out; hidden_sizes = trainoptions[:hidden], act = relu, fact=sigmoid)

    # Initial training (supervised)
    opt = trainoptions[:opt]
    _loss(x, y) = GoCompetition.loss(m, x, y)
    l = []
    θ = []
    function batchloss(data; test = false)
        test ? Flux.testmode!(m) : nothing;
        mean(map(data) do (X, y) _loss(X, y) end).data
    end
    do_earlystop(epochs, l) = epochs > 50 ? true : false # Put whatever function we want here
    do_stop(epochs, l) = epochs >= trainoptions[:nepochs] ? true : do_earlystop(epochs, l)
    cb = () -> (push!(l, [batchloss(data_train); batchloss(data_test, test = true)]);
                push!(θ, GoCompetition.get_w(m))
                )
    #cb() # Get pre-training performance
    epochs = 0
    ps = Flux.params(m)
    while !do_stop(epochs, l)
        @info("Epoch: $epochs")
        epochs += 1
        Flux.testmode!(m, false)
        for (idx, d) in enumerate(data_train)
            gs = gradient(ps) do
                _loss(d...)
            end
            update!(opt, ps, gs)
        end
        cb()
    end
    l = hcat(l...)'
    l = DataFrame(Dict(:training => l[:, 1], :testing => l[:, 2]))

    # Initialisation performance
    Flux.testmode!(m)
    post_sl = GoCompetition.evaluate(m, data_test, GoCompetition.ActiveSet) # PERFORMANCE ON ESTIMATING ACTIVE SET
    pre_meta = map(data_test) do (dt) metaevaluate(m, network, dt, ActiveSet; solver = solver) end
    pre_meta = vcat(pre_meta...)

    # Metatrain
    binding_template = create_bindingdict_template(data_train[1][2])
    _metaobjective(x) = metaobjective(x, m, network, data_train, identity, ActiveSet; solver = solver)
    res = optimize(_metaobjective, get_w(m), metaoptimizer, metaoptions)

    # Final performance
    post_meta = map(data_test) do (dt) metaevaluate(m, network, dt, ActiveSet; solver = solver) end
    post_meta = vcat(post_meta...)

    # Return
    return Dict(
        :trained_model => m, # Note that this will be metaoptimized
        :θ => θ,
        :train_metrics => l,
        :post_sl => post_sl,
        :pre_meta => pre_meta,
        :metaoptres => res,
        :post_meta => post_meta,
    )
end

function stopping(T=Float32; patience=10)
    best_loss::T = typemax(T)
    remaining_patience = patience
    function do_earlystop!(latest_loss)
        if latest_loss < best_loss
            remaining_patience = patience
            best_loss = latest_loss::T
            return false
        else
            remaining_patience -= 1
            if remaining_patience < 0
                return true
            else
                return false
            end
        end
    end
    return do_earlystop!
end
