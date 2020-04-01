using GoCompetition
using Ipopt
using JuMP
using PowerModels
using Random
using Flux
using Test
using MLDataUtils
using Statistics
using Optim
using OrderedCollections # Define sort on dict
using DataFrames
using MathOptInterface

# Disable PowerModels output
PowerModels.silence()

using GoCompetition: VerboseSolver

@testset "GoCompetition.jl" begin
    include("OPFSampler.jl")
    include("model.jl")
    include("binding.jl")
    include("distributed.jl")
end
