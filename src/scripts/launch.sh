export BS=10
export NP=10
export NI=50
date >> results_${NP}_${NI}.txt
julia --project=../../Project.toml  ./experiment.jl --casepath=/home/ubuntu/pglib-opf/pglib_opf_case162_ieee_dtc.m --datapath=/home/ubuntu/Cases_1000sam_load0.15_rest0.1/pglib_opf_case162_ieee_dtc_data.jld2 --P 8 --batchsize ${BS} --min_epochs 50 --metatrainsize 100 --PSOparticles ${NP} --PSOiterations ${NI} --lr 1e-4 >> results_${NP}_${NI}.txt
date >> results_${NP}_${NI}.txt
