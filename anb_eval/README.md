## ANB Evaluation
We offer NAS optimizers we utilized for uni- and bi-objective optimization, combined with the results, plots and searched models.

First install requirements using

``` bash
pip3 install -r requirements.txt
```

### Bi-objective optimization
#### Random Search
To perform bi-objective accuracy-throughput random search, use the following command (example for vck190, throughput. Please adapt according to device/metric need):

``` bash
python3 search_mo.py --arch_ep 250 --episodes 6 --device vck190 --metric throughput --algorithm RS --simulated
```

This would perform simulated bi-objective random search using `vck190` `throughput` surrogate.

The results are logged in `logs/simulated` directory. Pareto-optimal solutions are plotted and saved as png in the directory. Also, to obtain the pareto optimal set of solutions, go to the logs directory and perform the following operations to load the results pkl file:

``` python3
import pickle
res = pickle.load(open('pareto_designs249.pkl','rb'))
print(res)
``` 

This will print a dictionary of pareto-optimal solutions in the format: accuracy: [throughput, architecture]. E.g., 

``` python3
63.659610748291016: [3228.82421875, [['MB', 1, 3, 1, 32, 16, 2, False], ['MB', 1, 3, 2, 16, 24, 3, True], ['MB', 4, 3, 2, 24, 40, 2, True], ['MB', 1, 5, 2, 40, 80, 3, False], ['MB', 6, 5, 1, 80, 112, 2, True], ['MB', 6, 3, 2, 112, 192, 3, True], ['MB', 6, 3, 1, 192, 320, 3, True]]]
```

Where throughput is measured in images/sec, and architecture is formatted as follows (Please also see _Appendix. B_ for allowed values):

``` python3
[[ block 0 type (Always Mobile Bottleneck (MB)), block 0 expansion factor, block 0 kernel size, block 0 stride, block 0 input channels, block 0 output channels, block 0 number of layers, block 0 squeeze-excite state], ...]
```

#### REINFORCE
Use the following command to run REINFORCE in bi-objective setting:

``` bash
python3 search_mo.py --arch_ep 250 --episodes 6 --device vck190 --metric throughput --algorithm PG --simulated --target_biobj 3000
```

`target_biobj` controls the target throughput as stated in _Appendix. G_. The results are again logged in `logs/simulated` diretory in a similar format as above, however, only pkl files are generated. Please use the same code as above to decode the pkl files contents.


### Uni-objective optimization
#### Regularized Evolution
To perform RE, use the following command:

``` bash
cd nas_optimizers && python3 regularized_evolution.py
```

This will perform simulated RE using only the accuracy surrogate. The results will be stored in `nas_optimizers/logs` in the form of csv with filename `acc_trajectory_rea_1.csv` in the format `cycle number, best model acc`.

#### Random Search
To perform uni-objective RS, please execute the following command:

``` bash
python3 search_uo.py --arch_epochs 100 --episodes 6 --algorithm RS --simulated
```

Result csv file is generated in the `logs` directory and is formatted as follows:

``` python3
arch_epoch, top1_acc_best, top5_acc_epoch_avg, top1_acc_epoch_best, top1_acc_epoch_avg
```

#### REINFORCE
To perform uni-objective REINFORCE, please execute the following command:

``` bash
python3 search_uo.py --arch_epochs 100 --episodes 6 --algorithm PG --simulated
```

Result csv file has the following format:

``` python3
arch_epoch, top1_acc_best, ...
```

## No-proxy eval results
The resulting models from zero-cost REINFORCE-based bi-objective search (Section. 5.2) are offered:

#### FPGA models
| Target Device | DPU Arch | Model Name | Top-1 Accuracy Float | Top-1 Accuracy Quantized | E2E throughput (fps), Multi Thread, measured on target | Download Link |
| :--- |    :----:  |    :----:    |    :----:   |     :----:   | :----: |    ---: |
| VCK190 | DPUCVDX8G_ISA3_C32B6 | efficientnet-vck190-a | 77.568 | 76.966 | 2805.14 | [TODO]() |
| VCK190 | DPUCVDX8G_ISA3_C32B6 | efficientnet-vck190-b | 76.656 | 76.300 | 4015.04 | [TODO]() |
| ZCU102 | DPUCZDX8G_ISA1_B4096 | efficientnet-zcu102-a | 77.698 | 77.432 | 271.94 | [TODO]() |
| ZCU102 | DPUCZDX8G_ISA1_B4096 | efficientnet-zcu102-b | 76.602 | 76.314 | 398.54 | [TODO]() |
| ZCU102 | DPUCZDX8G_ISA1_B4096 | efficientnet-zcu102-c | 75.048 | 74.794 | 520.06 | [TODO]() |

#### TPU models
| Target Device | Model Name | Top-1 Accuracy Float | E2E throughput (fps) | Download Link |
| :--- |    :----:  |    :----:  |    :----:  |    ---:  |
| TPUv3 | efficientnet-tpuv3-a | 77.921 | 1010.304 | [TODO]() |
| TPUv3 | efficientnet-tpuv3-b | 77.348 | 1012.89 | [TODO]() |

#### GPU models
| Target Device | Model Name | Top-1 Accuracy Float | E2E throughput (fps) | Download Link |
| :--- |    :----:  |    :----:  |    :----:  |    ---:  |
| A100 | efficientnet-a100-a | 77.81 | 7734.21 | [TODO]() |
| A100 | efficientnet-a100-b | 76.51 | 11098.05 | [TODO]() |
| RTX3090 | efficientnet-3090-a | 77.358 | 5988.54 | [TODO]() |
| RTX3090 | efficientnet-3090-b | 76.510 | 7674.69 | [TODO]() |

