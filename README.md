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
# Specify sample instance
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

The accuracy surrogate instance can be created using `anb.ANBEnsemble('xgb')`. The throughput surrogate instance can be created as follows

``` python3
ensemble_inst_thr = anb.ANBEnsemble("xgb", device="tpuv2", metric="throughput")
```

The supported `device`s and their corresponding `metric`s are as follows:

``` python3
supported_metrics = {
            "3090": ["throughput"],
            "a100": ["throughput"],
            "tpuv2": ["throughput"],
            "tpuv3": ["throughput"],
            "zcu102": ["latency", "throughput"],
            "vck190": ["latency", "throughput"],
        }
```

A possible result of running `example.py` is follows. The result will be different each run owing to the random sampling of architecture. For the manually specified sample, the results would be the same each run:

``` python3
Mean Accuracy: [51.687283 65.736916]  # [Acc of sample 1, Acc of sample 2]
Std Acc: [0.13972819 0.1365913 ]
Mean Throughput: [1320.735   883.5296]  # [Throughput of sample 1, of sample 2] in images/sec
Std Thr: [7.4857635 8.001069 ]  # Standard deviation in throughput is measured in images/sec
```

Since we passed two samples to `.query` methods, we get their corresponding results in arrays, first element of the array corresponds to the result of the first sample. 

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

Example result of the above command using `seed=3` is as follows but will be different each run:

``` python3
train metrics: {'mae': 0.20502309085633813, 'mse': 0.06897010112105674, 'rmse': 0.2626215930213217, 'r2': 0.9937321545409954, 'kendall_tau': 0.9485453372279331, 'kendall_tau_2_dec': 0.9490165549382614, 'kendall_tau_1_dec': 0.9521390453390867, 'spearmanr': 0.9963628656408133}
valid metrics: {'mae': 0.32042279419469843, 'mse': 0.19266443301611308, 'rmse': 0.4
3893556818297724, 'r2': 0.9827909607086215, 'kendall_tau': 0.9246529281557777, 'kendall_tau_2_dec': 0.9252142975144962, 'kendall_tau_1_dec': 0.9283951048238933, 'spearmanr': 0.991347023504538}
test metrics {'mae': 0.3169779104452133, 'mse': 0.16787281044461427, 'rmse': 0.4097228458905047, 'r2': 0.9839661171659461, 'kendall_tau': 0.9170514701750982, 'kendall_tau_2_dec': 0.9175674110264382, 'kendall_tau_1_dec': 0.9204615872718475, 'spearmanr': 0.98994312462432}
```

To fit the _throughput XGB surrogate for ZCU102 FPGA_ on random train/val/test splits:
``` bash
python3 fit_model.py --dataset_root <path/to/extracted/dataset/> --model xgb_accel --device zcu102 --metric throughput --model_config_path ./configs/model_configs/gradient_boosting/xgb_accel_zcu102_throughput_configspace.json --data_config_path configs/data_configs/nb_fpga.json --log_dir experiments/ --seed <seed>
```

When fitting surrogates for throughput/latency, use models <xgb/lgb/sklearn_forest/svr/svr_nu>\_accel, combined with --device <zcu102/vck190/tpuv2/tpuv3/a100/3090> and --metric <throughput/latency>. Throughput is supported by all 6 devices while latency is supported by only the FPGAs.

The dataset splits utilized in this work were generated using the `create_data_splits.py` file. The splits are located in configs directory [here](https://github.com/afzalxo/Accel-NASBench/tree/master/configs/data_splits/default_split). Please place the dataset inside a directory structure specified inside the splits json files when training on the manual splits rather than random splits.

### Hyperparameter Optimization
The hyperparameters of the surrogates were optimized using [SMAC3](https://github.com/automl/SMAC3). Plase see `do_hpo.py` and `shell/hpo_all.sh` for details and file an issue if face an issue trying to perform HPO. 

The searched hyperparameters for various device/metric pairs can be found [here](https://github.com/afzalxo/Accel-NASBench/tree/master/configs/model_configs).

### Plots and Tables
The subdirectory plots contains the code and data for makings the plots and tables. It relies on the benchmark to generate the predictions. In order to make the plots, LGB surrogate model is needed and can be downloaded [here](https://figshare.com/ndownloader/files/40181317). Extract the LGB surrogate into the `anb_models_0_9` directory. Also needed are the ablation surrogate models. These are models that are trained on subsets of the total datasets, and are required to make e.g., Fig. 2 of the paper. These models can be downloaded [TODO](). Extract the ablation models in `experiments/ablation_models` since that is where the models are loaded from as follows:

``` python3
subdir = f"experiments/ablation_models/xgb/model_{q}samples-{seed}"  # q is the number of samples and seed is the seed with which the ablation model was trained.
```

After placing the LGB and ablation models in the appropriate directories, run the plotting script as follows

``` bash
python3 plots/ablations_plots/scatter_multiseed_eval.py
```

This would generate Fig. 1 and Fig. 2 plots and save pdfs of them.

### ANB Evaluation
Please see [anb-eval]()

### Dataset Collection Pipelines
Owing to the complex instrumentation of dataset collection, we have an entire repository that details collection pipelines for accuracy, throughput, and latency. Please see [ANB-DatasetCollection](https://anon-github.automl.cc/r/ANB-DatasetCollection-C564). Please note that collection of throughput/latency requires specialized hardware such as TPUs and FPGAs.

## Acknowledgements
This repository builds upon code from the following repositories:

- [NASBench-301](https://github.com/automl/nasbench301)
- [ConfigSpace](https://github.com/automl/ConfigSpace)
- [SMAC3](https://github.com/automl/SMAC3)

We are grateful to the authors of these repositories for their contributions.
