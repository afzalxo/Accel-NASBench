#!/bin/bash

model="svr"

command="python3 do_hpo.py --model $model --dataset_root ../dataset_parsers/datasets_final_jsons/json/ --model_config_path ./configs/model_configs/svr/svr_configspace.json --data_config_path ./configs/data_configs/nb_fpga.json"
echo "Running command: $command"
$command
rm -rf smac3_output

model="svr_nu"

command="python3 do_hpo.py --model $model --dataset_root ../dataset_parsers/datasets_final_jsons/json/ --model_config_path ./configs/model_configs/svr_nu/svr_nu_configspace.json --data_config_path ./configs/data_configs/nb_fpga.json"
echo "Running command: $command"
$command
rm -rf smac3_output

models=("svr" "svr_nu")
devs=("vck190" "zcu102" "3090" "a100" "tpuv2" "tpuv3")

for model in "${models[@]}"
do
	for dev in "${devs[@]}"
	do
	    if [[ $dev == "zcu102" || $dev == "vck190" ]]; then
		metrics=("throughput" "latency")
	    else
		metrics=("throughput")
	    fi
	    for metric in "${metrics[@]}"
	    do
		    command="python3 do_hpo.py --model ${model}_accel --dataset_root ../dataset_parsers/datasets_final_jsons/json/ --model_config_path ./configs/model_configs/${model}/${model}_configspace.json --data_config_path ./configs/data_configs/nb_fpga.json --device $dev --metric $metric"
		    echo "Running command: $command"
		    $command
		    rm -rf smac3_output
	    done
	done
done
