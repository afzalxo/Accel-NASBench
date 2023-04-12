#!/bin/bash

models=("svr_nu")

devs=("vck190" "zcu102" "3090" "a100" "tpuv2" "tpuv3")
seeds=(1 2 3 4 5 6)

for model in "${models[@]}"
do
	if [[ $model == "xgb" || $model == "lgb" ]]; then
		dir="gradient_boosting"
	elif [[ $model == "sklearn_forest" ]]; then
		dir="random_forest"
	elif [[ $model == "svr" ]]; then
		dir="svr"
	elif [[ $model == "svr_nu" ]]; then
		dir="svr_nu"
	fi
	for seed in "${seeds[@]}"
	do
		command="python3 fit_model.py --model $model --dataset_root ../dataset_parsers/datasets_final_jsons/json/ --model_config_path ./configs/model_configs/$dir/${model}_configspace.json --data_config_path ./configs/data_configs/nb_fpga.json --data_splits_root configs/data_splits/default_split/ --seed $seed"
		echo "Running command: $command"
		$command
	done
	for dev in "${devs[@]}"
	do
	    if [[ $dev == "zcu102" || $dev == "vck190" ]]; then
		metrics=("throughput" "latency")
	    else
		metrics=("throughput")
	    fi
	    for metric in "${metrics[@]}"
	    do
		for seed in "${seeds[@]}"
		do
		    command="python3 fit_model.py --model ${model}_accel --dataset_root ../dataset_parsers/datasets_final_jsons/json/ --model_config_path ./configs/model_configs/$dir/${model}_accel_${dev}_${metric}_configspace.json --data_config_path ./configs/data_configs/nb_fpga.json --device $dev --seed $seed --metric $metric --data_splits_root configs/data_splits/default_split/"
		    echo "Running command: $command"
		    $command
		done
	    done
	done
done
