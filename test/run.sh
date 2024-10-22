#!/bin/bash

# This is all based on the assumption you execute from test/

# We want values from 2^0 to 2^23, as 2^24 will not compile
n_start=0
n_end=23

# Generate array of world counts: world_counts=(2^n_start, 2^(n_start+1), ..., 2^n_end)
world_counts=()
for (( n=n_start; n<=n_end; n++ )); do
    world_counts+=( $((2**n)) )
done

echo "Simulating world counts: ${world_counts[@]}"

timestamp_hash=$(date +%s)

output_dir="./results/stdout"
mkdir -p $output_dir

# Prefer release but use debug otherwise
executable_path="../build/CUDASIMULATEWORLDS"

# Optional args for logging world, off for benchmark runs
world_log_idx=-1
simdata_output_dir="./results/simdata"

# Loop over each world count
for w in "${world_counts[@]}"; do
    output_file="${output_dir}/out_${w}.txt"

    echo "---------------------------" | tee -a $output_file
    echo "Running with $w Worlds:" | tee -a $output_file

    if [ "$world_log_idx" -ge 0 ]; then
        # Run the simulation with world logging enabled
        $executable_path $w $world_log_idx $simdata_output_dir >> $output_file 2>&1
    else
        # Run the simulation without world logging
        $executable_path $w >> $output_file 2>&1
    fi

    echo "Finished running with $w Worlds" | tee -a $output_file
done

echo "Finished all simulations"
