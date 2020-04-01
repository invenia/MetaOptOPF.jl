This package defines the necessary functions to run the experiments in https://arxiv.org/abs/1911.06784.

Full details can be seen in the paper, but it has these following conceptual components:
* `binding.jl`
- Functions for adding and removing from a JuMP certain constraints (PowerModels and JuMP):
* `model.jl`
- Neural network training blocks (Flux)
- Setting and getting neural network weights for interaction with Particle Swarm Optimization
* `powermodels.jl`
- Setting and getting grid parameters from a `powermodels` OPF formulation.
* `evaluate.jl`
- Helper functions used to evaluate the computational cost of running OPF, such as timing values.

The main experiments presented in the paper are executed through `./scripts/experiment.jl`.
This in turn can be launched with `launch.sh`.
These are the conceptual blocks in `experiment.jl` in the `_main` function:
* Take a model and train it 'classically' (i.e. with gradient descent)
- This is in the block `while !do_stop(epochs, l_val)`. Essentially training the NN with
early stopping with `update!(opt, ps, gs)`
* Defining a metaloss function: `function _metaloss(...)`
* Following this we then optimize, from this initialization, subject to the metaloss:
- `res = optimize(_metaloss, GoCompetition.get_w(model), ParticleSwarm...`
- Here for example `_metaloss` is a function that returns the metaloss, and is passed into
the optimizer. `get_w` gets the weights of the NN, and `ParticleSwarm` refers to the optimizer, etc.

Branches:
master: DC-OPF
ac-opf: AC-OPF
