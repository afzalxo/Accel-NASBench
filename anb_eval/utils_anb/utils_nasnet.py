import random
import numpy as np
import os
import shutil
import oapackage
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ConfigSpace.configuration_space import Configuration


def count_params(model):
    return sum(np.prod(v.shape) for name, v in model.named_parameters()) / 1e6


def pareto_frontier(datapoints, maxX=True, maxY=True):
    pareto = oapackage.ParetoDoubleLong()
    for ii in range(0, datapoints.shape[1]):
        w = oapackage.doubleVector((datapoints[1, ii], datapoints[0, ii]))
        pareto.addvalue(w, ii)
    pareto.show(verbose=1)
    lst = pareto.allindices()
    # print(lst)
    accs_pareto = datapoints[0, lst]
    size_pareto = datapoints[1, lst]
    # print(accs_pareto, size_pareto)
    return accs_pareto, size_pareto, lst


def plot_scatter_pareto(x_vals, y_vals, pareto, fname, label_top5=False, label_points=None, baseline_effnet=None):
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    if label_points is not None:
        fig = plt.figure(figsize=(9, 7))
    plt.clf()
    df = pd.DataFrame({"Throughput": x_vals, "Accuracy": y_vals})
    df_pareto = pd.DataFrame({"Paretox": pareto[0, :], "Paretoy": pareto[1, :]})
    sns.scatterplot(data=df, y="Accuracy", x="Throughput", marker="o", s=70)
    sns.lineplot(data=df_pareto, y="Paretoy", x="Paretox", color="green", markersize=20, linewidth=4)
    sns.scatterplot(data=df_pareto, y="Paretoy", x="Paretox", marker="o", color="olivedrab", s=200)
    if label_top5:
        colors = ["blue", "orange", "green", "red", "purple"]
        highest_y_indices = sorted(range(len(df_pareto["Paretoy"])), key=lambda i: df_pareto["Paretoy"][i])[-5:]
        highest_y_values = [df_pareto["Paretoy"][i] for i in highest_y_indices]
        highest_y_coords = list(zip([df_pareto["Paretox"][i] for i in highest_y_indices], highest_y_values))
        # label the points with the highest y values
        sca = sns.scatterplot(x=[df_pareto["Paretox"][i] for i in highest_y_indices], y=[df_pareto["Paretoy"][i] for i in highest_y_indices], color=colors, s=50)
        # legend_handles, legend_labels = sca.legend_elements(prop="colors", alpha=0.6, size=10, func=lambda x:highest_y_coords[colors.index(df_pareto["Paretox"])])
        legend_handles = [mpatches.Patch(color=color, label=f"{int(coord[0])}, {coord[1]:.3f}") for color, coord in zip(colors, highest_y_coords)]
        legend_labels = [handle.get_label() for handle in legend_handles]
        plt.legend(title="High-accuracy pareto solutions", handles=legend_handles, labels=legend_labels)# labels=[f"({int(coord[0])}, {coord[1]:.3f})" for coord in highest_y_coords])
        # for i, artist in enumerate(legend_handles[-len(highest_y_coords):]):
        #    artist.set_color(colors[i])
        # for coord in highest_y_coords:
        #    plt.annotate(f"({int(coord[0])}, {coord[1]:.3f})", xy=coord, xytext=(5, 5), textcoords='offset points', fontsize=5)
    if label_points is not None:
        xvals = [point[0] for point in label_points]
        yvals = [point[1] for point in label_points]
        # colors = ['purple', 'magenta', 'red']
        colors = ['magenta', 'red']
        sns.scatterplot(x=xvals, y=yvals, marker="*", color=colors[::-1], s=600, zorder=10)
        # for i in range(len(label_points)):
        #    plt.annotate(f"({int(xvals[i]):d}, {yvals[i]:.2f})", (xvals[i], yvals[i]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=14)
        # annotations = ['effnet-zcu102-c - Thr: 540, Acc: 65.12', 'effnet-zcu102-b - Thr: 399, Acc: 66.21', 'effnet-zcu102-a - Thr: 285, Acc: 66.84']
        # annotations = ['effnet-vck190-b - Thr: 3736, Acc: 66.36', 'effnet-vck190-a - Thr: 3302, Acc: 67.08']
        # annotations = ['effnet-tpuv3-b - Thr: 1465, Acc: 66.91', 'effnet-tpuv3-a - Thr: 1316, Acc: 67.30']
        # annotations = ['effnet-a100-b - Thr: 12057, Acc: 66.24', 'effnet-a100-a - Thr: 8011, Acc: 67.07']
        annotations = ['effnet-3090-b - Thr: 8079, Acc: 66,24', 'effnet-3090-a - Thr: 6000, Acc: 66.98']
        legend_elements = [plt.Line2D([0], [0], marker='*', markersize=20, color=color, label=annotation, linestyle='') for color, annotation in zip(colors[::-1], annotations[::-1])]
        plt.legend(handles=legend_elements, loc="lower left", fontsize=20)
    if baseline_effnet is not None:
        xval = baseline_effnet[0]
        yval = baseline_effnet[1]
        sns.scatterplot(x=[xval], y=[yval], marker="s", color="magenta", s=40, zorder=10)
        # plt.annotate(f"({int(xval):d}, {yval:.2f})", (xval, yval), textcoords="offset points", xytext=(0, -10), ha="center", fontsize=8)

    plt.xlabel("Throughput (images/sec)", fontsize=22)
    plt.ylabel("")
    # plt.ylabel("Top-1 Accuracy (%)", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.savefig(fname, dpi=400)


def CustomSearchable(e, k, la, se):
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    blocktypes = ["MB"]
    layer_confs = []
    bchoices = random.choices(blocktypes, k=num_blocks)
    kchoices = k
    echoices = e
    lchoices = la
    sechoices = se
    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
            sechoices[i],
        ]
        layer_confs.append(conf)
    return layer_confs


def actions_indices_to_config(action_indices, configspace):
    _exps = []
    _kernels = []
    _lay = []
    _se = []
    action_values = []
    e_vals = [1, 4, 6]
    k_vals = [3, 5]
    l_vals = [1, 2, 3, 4]
    se_vals = [True, False]
    block_iter = 0
    config_dict = {}
    for i, index in enumerate(action_indices):
        if i % 4 == 0:
            action_values.append(e_vals[index])
            config_dict[f"block{block_iter}_e"] = e_vals[index]
            _exps.append(e_vals[index])
        elif (i - 1) % 4 == 0:
            action_values.append(k_vals[index])
            config_dict[f"block{block_iter}_k"] = k_vals[index]
            _kernels.append(k_vals[index])
        elif (i - 2) % 4 == 0:
            action_values.append(l_vals[index])
            config_dict[f"block{block_iter}_l"] = l_vals[index]
            _lay.append(l_vals[index])
        elif (i - 3) % 4 == 0:
            action_values.append(se_vals[index])
            config_dict[f"block{block_iter}_se"] = se_vals[index]
            _se.append(se_vals[index])
        else:
            raise ValueError("Unreachable...")
        if (i + 1) % 4 == 0:
            block_iter += 1
    config = Configuration(configspace, values=config_dict)
    return config, CustomSearchable(_exps, _kernels, _lay, _se)


def sort_dict_by_key(_dict):
    ret = dict()
    for k in sorted(_dict):
        ret[k] = _dict[k]
    return ret


def configuration_to_searchable(config):
    config_dict = config.get_dictionary()
    _exps, _kernels, _lay, _se = dict(), dict(), dict(), dict()
    for k, v in config_dict.items():
        block_num = int(k.split("_")[0][-1])
        if "_e" in k:
            _exps[block_num] = int(v)
        elif "_k" in k:
            _kernels[block_num] = int(v)
        elif "_l" in k:
            _lay[block_num] = int(v)
        elif "_se" in k:
            _se[block_num] = v

    _exps, _kernels, _lay, _se = (
        sort_dict_by_key(_exps),
        sort_dict_by_key(_kernels),
        sort_dict_by_key(_lay),
        sort_dict_by_key(_se),
    )
    return CustomSearchable(_exps, _kernels, _lay, _se)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)
