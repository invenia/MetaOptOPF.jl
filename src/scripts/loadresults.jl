using GoCompetition
using Flux
using JLD2
using BSON
using PowerModels




casepath = "/Users/alexr/Projects/GoCompetition/pglib-opf/pglib_opf_case162_ieee_dtc.m"
@load "data_test.jld2"

network = PowerModels.parse_file(casepath)
GoCompetition.grid_dcopf_cleanup!(network)
setting = Dict("output" => Dict("branch_flows" => true))
pm_model = build_model(network, DCPPowerModel, PowerModels.post_opf, setting=setting);
x_orig = GoCompetition.get_x(network)
x_orig[x_orig .== 0] .= 1.0
transform = x -> (x - x_orig) ./ x_orig

size_in = size(data_test[1][1], 1)
size_out = GoCompetition.totalconstraints(pm_model.model)

model_pre = Chain(Dense(size_in, 50), BatchNorm(50, relu), Dropout(0.4),
            Dense(50, 50), BatchNorm(50, relu), Dropout(0.4),
            Dense(50, size_out, σ))
model_post = deepcopy(model_pre)


BSON.@load "model_pre.bson" weights_pre
BSON.@load "model_post.bson" weights_post

Flux.loadparams!(model_pre, weights_pre)
