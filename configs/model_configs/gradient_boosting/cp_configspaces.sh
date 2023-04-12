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
	    cp lgb_configspace.json lgb_accel_${dev}_${metric}_configspace.json
    done
done
