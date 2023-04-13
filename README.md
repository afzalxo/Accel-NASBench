# Accel-NASBench: A Surrogate Benchmark for Accelerator-aware NAS
TLDR: We offer a surrogate NAS benchmark for the ImageNet2012 dataset on the MnasNet search space (Please see the specification of the search space in Appendix B of the paper and [here](https://github.com/afzalxo/Accel-NASBench/blob/master/configurationspaces/configuration_space_writer.py)). We also offer inference throughput surrogates for 6 hardware accelerators: Cloud TPUv2 and TPUv3, A100 and RTX3090 GPUs, Xilinx Ultrascale+ ZCU102 and Versal AI Core VCK190 FPGAs, and latency surrogates for the FPGA platforms. The benchmark allows evaluation without model and dataset proxies and can be utilized for benchmarking discrete NAS optimizers.

The XGB surrogates are available on figshare [here](https://figshare.com/ndownloader/files/40109821). They will be automatically downloaded if you run the example.py file using `python3 example.py`. 

To install the requirements, please clone the repository and run the following command inside the cloned directory

``` bash
pip3 install -r requirements.txt
```

Then run the example using

``` bash
python3 example.py
```

The example file will download the XGB surrogates to `anb_models_0_9` directory in the project directory. The surrogates allow evaluation of accuracy, throughput on the 6 accelerators, and latency on the FPGAs. Search space sample can be manually specified using

``` python3
from configurationspaces.searchspaces import EfficientNetSS as ss
# Create search space instance
search_space = ss()
# Specify instance
test_sample_man = search_space.manual_sample(
    [
        [1, 6, 6, 6, 6, 6, 6],  # Expansion Factor for the 7 blocks
        [3, 3, 5, 3, 5, 5, 3],  # Kernel Sizes
        [1, 2, 2, 3, 3, 4, 1],  # Number of Layers in block
        [True, True, True, True, True, True, True],   # Squeeze-Excite state
    ]
)
```

or sample 4 random architectures using

``` python3
test_samples_rand = search_space.random_sample(4)
```

### Accel-NASBench dataset
Dataset utilized to train the surrogates is provided in json format [here](https://github.com/afzalxo/Accel-NASBench/tree/master/anb_dataset) similar to that used by NASBench-301. Please see a sample result_x.json file to understand its different fields. Each result_x.json file contains architecture specification, accuracy, train time, and all on-device throughput/latency measurement mean and standard deviations. We train the surrogates using mean throughput/latency values. Accuracy is evaluated only at a single seed.

### Fit surrogates using the dataset
Although we provide the surrogate models, they can be trained manually using the dataset provided. Please follow the following steps in order to train the surrogates.
1. Download and extract the anb_dataset_jsons.tar.gz archive.
2. Take note of the root directory of the extracted dataset.
3. Run the following command inside the project directory to fit the _accuracy XGB surrogate_ on random train/val/test splits of 0.8/0.1/0.1 ratio
``` bash
python3 fit_model.py --dataset_root <path/to/extracted/dataset/> --model xgb --model_config_path ./configs/model_configs/gradient_boosting/xgb_configspace.json --data_config_path configs/data_configs/nb_fpga.json --log_dir experiments/ --seed <seed>
```

Or to fit the _throughput XGB surrogate for ZCU102 FPGA_ on random train/val/test splits:
``` bash
python3 fit_model.py --dataset_root <path/to/extracted/dataset/> --model xgb_accel --device zcu102 --metric throughput --model_config_path ./configs/model_configs/gradient_boosting/xgb_accel_zcu102_throughput_configspace.json --data_config_path configs/data_configs/nb_fpga.json --log_dir experiments/ --seed <seed>
```

The dataset splits utilized in this work were generated using the `create_data_splits.py` file. The splits are located in configs directory [here](https://github.com/afzalxo/Accel-NASBench/tree/master/configs/data_splits/default_split). Please place the dataset inside a directory structure specified inside the splits json files when training on the manual splits rather than random splits.

## Acknowledgements
This repository builds upon code from the following repositories:

- [NASBench-301](https://github.com/automl/nasbench301)
- [ConfigSpace](https://github.com/automl/ConfigSpace)
- [SMAC3](https://github.com/automl/SMAC3)

We are grateful to the authors of these repositories for their contributions.

### Dataset Collection Pipelines
Owing to the complex instrumentation of dataset collection, we have an entire repository that details collection pipelines for accuracy, throughput, and latency. Please see [ANB-DatasetCollection](https://github.com/afzalxo/ANB-DatasetCollection). Please note that collection of throughput/latency requires specialized hardware such as TPUs and FPGAs.
