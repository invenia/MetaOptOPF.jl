export BS=100
export NP=10
export NI=50
export MIN_EPOCHS=50
export MTS=100
export PATIENCE=10
export P=16
export DATAPATH="/home/ubuntu/samples/Cases_10000sam_load0.15_rest0.1/pglib_opf_case300_ieee_data.jld2"
export CASEPATH="/home/ubuntu//pglib-opf/pglib_opf_case300_ieee.m"
export N=20
export PENALTY=2.0
julia --project=../../Project.toml  ./experiment.jl --casepath=${CASEPATH} --datapath=${DATAPATH} --P ${P} --batchsize ${BS} --min_epochs ${MIN_EPOCHS} --metatrainsize ${MTS} --PSOparticles ${NP} --PSOiterations ${NI} --lr 1e-4 --patience ${PATIENCE} --N ${N} --penalty_factor ${PENALTY}
# Launch with ./launch.jl | tee output.info
