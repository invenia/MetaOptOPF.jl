export BS=10
export NP=3
export NI=1
export MIN_EPOCHS=5
export MTS=5
export PATIENCE=0
export P=0
export DATAPATH="/Users/alexr/Projects/GoCompetition/samples/Cases_10000sam_load0.15_rest0.1/pglib_opf_case162_ieee_dtc_data.jld2"
export CASEPATH="/Users/alexr/Projects/GoCompetition/pglib-opf/pglib_opf_case162_ieee_dtc.m"
julia --project=../../Project.toml  ./experiment.jl --casepath=${CASEPATH} --datapath=${DATAPATH} --P ${P} --batchsize ${BS} --min_epochs ${MIN_EPOCHS} --metatrainsize ${MTS} --PSOparticles ${NP} --PSOiterations ${NI} --lr 1e-4 --patience ${PATIENCE}
