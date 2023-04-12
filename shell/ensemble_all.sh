#!/bin/bash

devs=("vck190" "zcu102" "3090" "a100" "tpuv2" "tpuv3")

for dev in "${devs[@]}"
do
    if [[ $dev == "zcu102" || $dev == "vck190" ]]; then
        metrics=("throughput" "latency")
    else
        metrics=("throughput")
    fi
    for metric in "${metrics[@]}"
    do
            command="python3 eval_ensemble.py --model svr_nu_accel --device $dev --metric $metric"
            echo "Running command: $command"
            $command
    done
done
