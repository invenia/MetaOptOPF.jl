using Pkg
Pkg.activate("../../")
using GoCompetition
using PowerModels
using JuMP
using Ipopt
using Memento
using LinearAlgebra
using FileIO
using JLD2
using MathOptInterface
using Random
Memento.config!("error")
PowerModels.silence()

job_num = "1"; # choose a unique number for the instance you are running so that the output is not overwitten by other instances
range = MersenneTwister(12345); # choose a unique integer number for random seed initializer so that the samples of this instance is different from another instance
case_path = "/home/ubuntu/cases/pglib_opf_case9241_pegase.m";
case_name = "pglib_opf_case9241_pegase";
num_samples = 650; # number of samples that you want from sampler


network = PowerModels.parse_file(case_path);
pm_model = deepcopy(network);
GoCompetition.grid_dcopf_cleanup!(pm_model)

par = Dict("case_network" => pm_model, "dev_load_pd" => 0.15,
           "dev_gen_max" => 0.1, "dev_rate_a" => 0.1, "dev_br_x" => 0.1);

cases = RunDCSampler(num_samples, par; rng = range);

save(case_name * job_num * ".jld2", Dict("cases" => cases)); # save `cases`

data, Flag_c = GoCompetition.case_to_data(cases; t = GoCompetition.ActiveSet);
save(case_name * job_num * "_data.jld2", Dict("data" => data)); # save `data`
