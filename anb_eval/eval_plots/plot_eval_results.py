# This file plots the scatters in Fig. 5 of the paper.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# plt.grid(linestyle='--', alpha=0.5)
sns.set_style("darkgrid")

# FPGA accuracies/throughputs for existing models obtained from
# Xilinx Vitis-AI model zoo at the following hyperlink
# https://xilinx.github.io/Vitis-AI/docs/reference/ModelZoo_VAI3.0_Github_web.htm

float_acc = [
    0.799,
    0.761,
    0.7702,
    0.6976,
    0.7399,
    0.7201,
    0.7227,
    0.7102,
    0.7013,
    0.7411,
    0.764,
    0.7681,
    0.752,
    0.7089,
    0.769,
    0.7501,
    0.7005,
    0.6756,
]

quant_acc = [
    0.789,
    0.76,
    0.766,
    0.6794,
    0.7331,
    0.6489,
    0.6775,
    0.678,
    0.6767,
    0.7194,
    0.756,
    0.7463,
    0.7436,
    0.7069,
    0.7515,
    0.7444,
    0.5603,
    0.6536,
]

throughput_zcu102 = [
    99.23,
    189.67,
    318.97,
    503.71,
    247.54,
    735.87,
    597.48,
    1038.05,
    764.41,
    504.48,
    122.51,
    84.68,
    214.04,
    43.40,
    156.48,
    580.00,
    1017.95,
    1073.70,
]

latency_zcu102 = [
    20.84,
    12.51,
    8.69,
    5.19,
    10.69,
    3.81,
    4.61,
    3.02,
    3.70,
    5.22,
    21.23,
    31.24,
    11.08,
    48.77,
    12.83,
    4.94,
    3.08,
    2.91,
]

throughput_vck190 = [
    1174.91,
    2688.35,
    2167.06,
    2436.64,
    1184.49,
    4887.82,
    4790.46,
    4935.86,
    4930.31,
    3680.09,
    1727.18,
    1200.09,
    3022.13,
    621.19,
    1810.3,
    4406.25,
    4941.36,
    4904.55,
]

model_name = [
    "ofa-rn50-224",
    "pt_resnet50",
    "effnet-edgetpu-s",
    "inceptionv1",
    "inceptionv2",
    "mobilenet-edge-0.75",
    "mobilenet-edge-1.0",
    "tf_mobilenetv1",
    "mobilenetv2_1.0",
    "mobilenetv2_1.4",
    "resnetv1_101",
    "resnetv1_152",
    "resnetv1_50",
    "vgg16",
    "effnet-b0",
    "effnet-lite",
    "tf2_mobilenetv1",
    "mobilenetv3_small",
]

design_mode = [
    "Searched",
    "Handcrafted",
    "Searched",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Searched",
    "Searched",
    "Handcrafted",
    "Searched",
]

my_models = [
    "effnet-zcu102-a",
    "effnet-zcu102-b",
    "effnet-zcu102-c",
    "effnet-vck190-a",
    "effnet-vck190-b",
]

my_models_lat_zcu = [
    "effnet-zcu102-m",
    "effnet-zcu102-l",
]

my_latency_zcu102 = [7.30, 9.07]
my_acc_q_zcu102_lat = [77.356, 77.52]

my_throughput_zcu102 = [271.94, 398.54, 520.06]
my_acc_zcu102 = [77.698, 76.602005, 75.04799652]
my_acc_q_zcu102 = [77.432, 76.314, 74.794]
my_throughput_vck190 = [2802.54, 4015.04]
my_acc_vck190 = [77.56800079, 76.6559906]
my_acc_q_vck190 = [76.966, 76.3]

marker_dict = {"Searched": "o", "Handcrafted": "o"}

df = pd.DataFrame(
    {
        "float_acc": list(np.array(float_acc) * 100.0) + my_acc_zcu102,
        "throughput_zcu102": throughput_zcu102 + my_throughput_zcu102,
        "model_name": model_name + my_models[:3],
        "design_mode": design_mode + ["Searched", "Searched", "Searched"],
    }
)

sns.scatterplot(
    data=df[df["float_acc"] > 73.0],
    x="throughput_zcu102",
    y="float_acc",
    hue="design_mode",
    style="design_mode",
    markers=marker_dict,
    alpha=0.7,
)

# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.2
    fontsize = 18
    if row["model_name"] == "resnetv1_152":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = 0
    elif row["model_name"] == "resnetv1_101":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = 0
    elif row["model_name"] == "effnet-b0":
        ha = "right"
        va = "bottom"
        offset_y = +0.05
        offset_x = -5
    elif "zcu102" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10
    elif "edgetpu-s" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10
        fontsize = 16
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10

    plt.text(
        row["throughput_zcu102"] + offset_x,
        row["float_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

# Set axis labels and title
plt.xlabel("Throughput ZCU102 (images/sec)")
plt.ylabel("Top-1 Float Accuracy")
plt.title("Accuracy vs Throughput on ZCU102")
plt.savefig("plot_results_zcu102.pdf")
plt.clf()


owner = [
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Ours",
    "Ours",
    "Ours",
]

df = pd.DataFrame(
    {
        "quant_acc": list(np.array(quant_acc) * 100.0) + my_acc_q_zcu102,
        "throughput_zcu102": throughput_zcu102 + my_throughput_zcu102,
        "model_name": model_name + my_models[:3],
        "design_mode": design_mode + ["Searched", "Searched", "Searched"],
        "owner": owner,
    }
)

ax = sns.scatterplot(
    data=df[df["quant_acc"] > 70.0][df["owner"] == "Previous"],
    x="throughput_zcu102",
    y="quant_acc",
    # hue="design_mode",
    # style="design_mode",
    markers=marker_dict,
    s=85,
    alpha=0.7,
)
sns.lineplot(
    data=df[df["quant_acc"] > 70.0][df["owner"] == "Ours"],
    x="throughput_zcu102",
    y="quant_acc",
    linestyle="--",
    linewidth=1,
    color="red",
    marker="o",  # marker_dict,
    markersize=11,
    alpha=0.7,
    ax=ax,
)
# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.2
    fontsize = 13
    if row["model_name"] == "resnetv1_152":
        ha = "center"
        va = "bottom"
        offset_y = -0.5
        offset_x = 0
    elif row["model_name"] == "resnetv1_101":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -6
        fontsize = 11
    elif row["model_name"] == "effnet-b0":
        ha = "center"
        va = "top"
        offset_y = -0.1
        offset_x = 0
    elif "zcu102" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10
    elif "edgetpu-s" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.2
        offset_x = -10
        fontsize = 12
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10
    elif row["model_name"] == "pt_resnet50":
        offset_x = +20
    elif row["model_name"] == "effnet-lite":
        offset_x = -30

    plt.text(
        row["throughput_zcu102"] + offset_x,
        row["quant_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

plt.subplots_adjust(left=0.18, right=0.85, bottom=0.15, top=0.90)
# Set axis labels and title
plt.xlabel("Throughput ZCU102 (images/sec)", fontsize=20)
plt.ylabel("Top-1 Quantized Accuracy (%)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("plot_results_zcu102_quantized.pdf")
plt.clf()

### Scatter quant zcu102 latency
owner = [
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Previous",
    "Ours",
    "Ours",
    "Ours",
]

my_models_lat_zcu = [
    "effnet-zcu102-m",
    "effnet-zcu102-l",
]
my_latency_zcu102 = [7.30, 9.07]
my_acc_q_zcu102_lat = [77.356, 77.52]
df = pd.DataFrame(
    {
        "quant_acc": list(np.array(quant_acc) * 100.0) + my_acc_q_zcu102_lat,
        "lat_zcu102": latency_zcu102 + my_latency_zcu102,
        "model_name": model_name + my_models_lat_zcu,
        "design_mode": design_mode + ["Searched", "Searched"],
        "owner": owner[:-1],
    }
)

ax = sns.scatterplot(
    data=df[df["quant_acc"] > 70.0][df["owner"] == "Previous"],
    x="lat_zcu102",
    y="quant_acc",
    # hue="design_mode",
    # style="design_mode",
    markers=marker_dict,
    s=85,
    alpha=0.7,
)
sns.lineplot(
    data=df[df["quant_acc"] > 70.0][df["owner"] == "Ours"],
    x="lat_zcu102",
    y="quant_acc",
    linestyle="--",
    linewidth=1,
    color="red",
    marker="o",  # marker_dict,
    markersize=11,
    alpha=0.7,
    ax=ax,
)
# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.2
    fontsize = 13
    if row["model_name"] == "resnetv1_152":
        ha = "center"
        va = "bottom"
        offset_y = -0.7
        offset_x = 0
    elif row["model_name"] == "resnetv1_101":
        ha = "left"
        va = "center"
        offset_y = 0
        offset_x = +0.7
        fontsize = 11
    elif row["model_name"] == "effnet-b0":
        ha = "center"
        va = "top"
        offset_y = -0.1
        # offset_x = 0
    elif "zcu102-m" in row["model_name"]:
        ha = "center"
        va = "top"
        offset_y = -0.1
        offset_x = +0.0
        fontsize = 11
    elif "zcu102-l" in row["model_name"]:
        ha = "left"
        va = "center"
        offset_y = +0.2
        offset_x = +0.5
        fontsize = 11
    elif "edgetpu-s" in row["model_name"]:
        ha = "left"
        va = "center"
        offset_y = -0.2
        offset_x = +0.7
        fontsize = 11
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        # offset_x = -10
    elif "mobilenetv2_1.4" in row["model_name"]:
        ha = "left"
        va = "center"
        offset_y = 0
        offset_x = +0.6
    elif row["model_name"] == "pt_resnet50":
        # offset_x = +20
        fontsize = 11
    elif row["model_name"] == "effnet-lite":
        #offset_x = -30
        va = "bottom"
        fontsize = 11

    plt.text(
        row["lat_zcu102"] + offset_x,
        row["quant_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

plt.subplots_adjust(left=0.18, right=0.85, bottom=0.15, top=0.90)
# Set axis labels and title
plt.xlabel("Latency ZCU102 (ms)", fontsize=20)
plt.ylabel("Top-1 Quantized Accuracy (%)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("plot_results_zcu102_quantized_latency.pdf")
plt.clf()
### End latency zcu102 here

df = pd.DataFrame(
    {
        "float_acc": list(np.array(float_acc) * 100.0) + my_acc_vck190,
        "throughput_vck190": throughput_vck190 + my_throughput_vck190,
        "model_name": model_name + my_models[3:],
        "design_mode": design_mode + ["Searched", "Searched"],
    }
)

sns.scatterplot(
    data=df[df["float_acc"] > 73.0],
    x="throughput_vck190",
    y="float_acc",
    hue="design_mode",
    style="design_mode",
    markers=marker_dict,
    alpha=0.7,
)

# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.2
    fontsize = 15
    if row["model_name"] == "resnetv1_152":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = 0
    elif row["model_name"] == "resnetv1_101":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = 0
    elif row["model_name"] == "effnet-b0":
        ha = "right"
        va = "bottom"
        offset_y = +0.03
        offset_x = -30
    elif "vck190" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -80
    elif "edgetpu-s" in row["model_name"]:
        ha = "center"
        va = "bottom"
        offset_y = -0.4
        offset_x = 0
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -80
    elif "mobilenetv2_1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = -5
    elif "mobilenet-edge-1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = 0

    plt.text(
        row["throughput_vck190"] + offset_x,
        row["float_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

# Set axis labels and title
plt.xlabel("Throughput VCK190 (images/sec)")
plt.ylabel("Top-1 Float Accuracy")
plt.title("Accuracy vs Throughput on VCK190")
plt.savefig("plot_results_vck190.pdf")
plt.clf()

df = pd.DataFrame(
    {
        "quant_acc": list(np.array(quant_acc) * 100.0) + my_acc_q_vck190,
        "throughput_vck190": throughput_vck190 + my_throughput_vck190,
        "model_name": model_name + my_models[3:],
        "design_mode": design_mode + ["Searched", "Searched"],
        "owner": owner[:-1],
    }
)

ax = sns.scatterplot(
    data=df[df["quant_acc"] > 70.0][df["owner"] == "Previous"],
    x="throughput_vck190",
    y="quant_acc",
    # hue="owner",
    # style="design_mode",
    markers=marker_dict,
    s=85,
    alpha=0.7,
)
sns.lineplot(
    data=df[df["quant_acc"] > 70.0][df["owner"] == "Ours"],
    x="throughput_vck190",
    y="quant_acc",
    # hue="owner",
    # style="design_mode",
    linestyle="--",
    linewidth=1,
    color="red",
    marker="o",  # marker_dict,
    markersize=11,
    alpha=0.7,
    ax=ax,
)


# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.2
    fontsize = 15
    if row["model_name"] == "resnetv1_152":
        ha = "center"
        va = "top"
        offset_y = -0.1
        offset_x = 0
        fontsize = 12
    elif row["model_name"] == "resnetv1_101":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -50
        fontsize = 15
    elif row["model_name"] == "effnet-b0":
        ha = "center"
        va = "top"
        offset_y = -0.05
        offset_x = 0
    elif "vck190" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = -0
        offset_x = -50
    elif "edgetpu-s" in row["model_name"]:
        ha = "right"
        va = "bottom"
        offset_y = -0.5
        offset_x = 0
        fontsize = 12
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -80
    elif "mobilenetv2_1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = -5
    elif "mobilenet-edge-1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = 0
    elif row["model_name"] == "vgg16":
        ha = "left"
        va = "center"
        offset_y = +0.08
        offset_x = +50
    elif row["model_name"] == "effnet-lite":
        ha = "right"
        va = "center"
        offset_y = +0.1
        offset_x = -60

    plt.text(
        row["throughput_vck190"] + offset_x,
        row["quant_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

# Set axis labels and title
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.13, top=0.9)
plt.xlabel("Throughput VCK190 (images/sec)", fontsize=20)
# plt.ylabel("Top-1 Quantized Accuracy", fontsize=16)
plt.ylabel("")
plt.xticks(fontsize=13)
plt.yticks(fontsize=16)
plt.savefig("plot_results_vck190_quantized.pdf")
plt.clf()


design_modes = [
    "Searched",
    "Searched",
    "Searched",
    "Searched",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Searched",
    "Searched",
]
model_names = [
    "effnet-b0",
    "effnet-edgetpu-s",
    "effnet-lite",
    "mobilenetv3large",
    "mobilenetv2",
    "vgg16",
    "vgg19",
    "resnet34",
    "resnet50",
    "effnet-a100-a",
    "effnet-a100-b",
]
model_accs = [
    77.3,
    77.23,
    75.1,
    75.766,
    75.036,
    71.594,
    72.368,
    73.312,
    76.138,
    77.81,
    76.5039978,
]
throughput_a100 = [
    7414.006565,
    6875.408697,
    9851.583038,
    12710.82883,
    7817.801028,
    2446.885668,
    2088.77212,
    9775.561125,
    4517.213535,
    7734.210916,
    11098.04449,
]

df = pd.DataFrame(
    {
        "float_acc": model_accs,
        "throughput_a100": throughput_a100,
        "model_name": model_names,
        "design_mode": design_modes,
    }
)

ax = sns.scatterplot(
    data=df,
    x="throughput_a100",
    y="float_acc",
    # hue="design_mode",
    # style="design_mode",
    markers=marker_dict,
    s=85,
    alpha=0.7,
)
sns.lineplot(
    data=df[df["model_name"].str.contains("a100")],
    x="throughput_a100",
    y="float_acc",
    linestyle="--",
    linewidth=1,
    color="red",
    marker="o",  # marker_dict,
    markersize=11,
    alpha=0.7,
    ax=ax,
)
# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.15
    fontsize = 15
    if row["model_name"] == "resnetv1_101":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = 0
    elif row["model_name"] == "effnet-b0":
        ha = "center"
        va = "top"
        offset_y = -0.15
        offset_x = +800
    elif "vck190" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -25
    elif "edgetpu-s" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.01
        offset_x = -150
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10
    elif "mobilenetv2_1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = -5
    elif "mobilenet-edge-1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = 0
    elif row["model_name"] == "mobilenetv3large":
        ha = "right"
        va = "center"
        offset_x = -80
    elif row["model_name"] == "effnet-lite":
        ha = "left"
    elif row["model_name"] == "vgg16":
        ha = "left"
        offset_x = +200
        offset_y = +0.1
    elif row["model_name"] == "effnet-a100-a":
        ha = "right"
        offset_x = -300
        offset_y = +0.1

    plt.text(
        row["throughput_a100"] + offset_x,
        row["float_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

# Set axis labels and title
plt.subplots_adjust(left=0.25, right=0.85, bottom=0.13, top=0.9)
plt.xlabel("Thr. A100 (images/sec)", fontsize=20)
plt.ylabel("")
# plt.ylabel("Top-1 Float Accuracy", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=16)
plt.savefig("plot_results_a100.pdf")
plt.clf()


design_modes = [
    "Searched",
    "Searched",
    "Searched",
    "Searched",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Handcrafted",
    "Searched",
    "Searched",
]
model_names = [
    "effnet-b0",
    "effnet-edgetpu-s",
    "effnet-lite",
    "mobilenetv3large",
    "mobilenetv2",
    "vgg16",
    "vgg19",
    "resnet34",
    "resnet50",
    "effnet-rtx3090-a",
    "effnet-rtx3090-b",
]
model_accs = [
    77.3,
    77.23,
    75.1,
    75.766,
    75.036,
    71.594,
    72.368,
    73.312,
    76.138,
    77.358,
    76.5039978,
]
throughput_3090 = [
    5622.990996,
    4287.88857,
    6576.863873,
    8627.44587,
    5099.790432,
    1189.548575,
    992.9753166,
    4576.825361,
    2580.44939,
    5988.538118,
    7674.689625,
]

df = pd.DataFrame(
    {
        "float_acc": model_accs,
        "throughput_3090": throughput_3090,
        "model_name": model_names,
        "design_mode": design_modes,
    }
)

ax = sns.scatterplot(
    data=df,
    x="throughput_3090",
    y="float_acc",
    # hue="design_mode",
    # style="design_mode",
    markers=marker_dict,
    s=85,
    alpha=0.7,
)
sns.lineplot(
    data=df[df["model_name"].str.contains("3090")],
    x="throughput_3090",
    y="float_acc",
    linestyle="--",
    linewidth=1,
    color="red",
    marker="o",  # marker_dict,
    markersize=11,
    alpha=0.7,
    ax=ax,
)
# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.15
    fontsize = 15
    if row["model_name"] == "resnetv1_101":
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = 0
    elif row["model_name"] == "effnet-b0":
        ha = "right"
        va = "bottom"
        offset_y = +0.06
        offset_x = -60
    elif "vck190" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -25
    elif "_small" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -10
    elif "mobilenetv2_1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = -5
    elif "mobilenet-edge-1.0" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = +0.03
        offset_x = 0
    elif row["model_name"] == "effnet-rtx3090-a":
        ha = "left"
    elif row["model_name"] == "effnet-edgetpu-s":
        ha = "right"
        va = "top"
        offset_y = -0.02
        offset_x = -50
    elif row["model_name"] == "effnet-lite":
        ha = "left"
    elif row["model_name"] == "mobilenetv3large":
        ha = "right"
        va = "center"
        offset_x = -200
        offset_y = +0.05
    elif row["model_name"] == "vgg16":
        ha = "left"
        va = "center"
        offset_y = +0.05
        offset_x = 100

    plt.text(
        row["throughput_3090"] + offset_x,
        row["float_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

# Set axis labels and title
plt.subplots_adjust(left=0.25, right=0.85, bottom=0.13, top=0.9)
plt.xlabel("Thr. RTX3090 (images/sec)", fontsize=20)
# plt.ylabel("Top-1 Float Accuracy", fontsize=14)
plt.ylabel("")
plt.xticks(fontsize=12)
plt.yticks(fontsize=16)
# plt.title("Accuracy vs Throughput on RTX3090")
plt.savefig("plot_results_3090.pdf")
plt.clf()


design_modes = [
    "Searched",
    "Searched",
    "Searched",
    "Searched",
    "Handcrafted",
    "Handcrafted",
    "Searched",
    "Searched",
]
model_names = [
    "effnet-b0",
    "effnet-edgetpu-s",
    "effnet-lite",
    "mobilenetv3large",
    "resnet34",
    "resnet50",
    "effnet-tpuv3-a",
    "effnet-tpuv3-b",
]
model_accs = [
    77.3,
    77.23,
    75.1,
    75.766,
    73.312,
    76.138,
    77.922,
    77.348,
]
throughput_tpuv3 = [
    887.5870738,
    1020.650326,
    992.1716222,
    1011.541685,
    995.4506631,
    1007.9266,
    1010.304076,
    1012.89,
]

df = pd.DataFrame(
    {
        "float_acc": model_accs,
        "throughput_tpuv3": throughput_tpuv3,
        "model_name": model_names,
        "design_mode": design_modes,
    }
)

ax = sns.scatterplot(
    data=df,
    x="throughput_tpuv3",
    y="float_acc",
    # hue="design_mode",
    # style="design_mode",
    markers=marker_dict,
    s=85,
    alpha=0.7,
    legend=False,
)
sns.lineplot(
    data=df[df["model_name"].str.contains("tpuv3")],
    x="throughput_tpuv3",
    y="float_acc",
    linestyle="--",
    linewidth=1,
    color="red",
    marker="o",  # marker_dict,
    markersize=11,
    alpha=0.7,
    ax=ax,
)
# Add data labels
vertical_offsets = [-0.3]
for i, row in df.iterrows():
    ha = "center"
    va = "top"
    offset_x = 0
    offset_y = -0.1
    fontsize = 15
    if "tpuv3" in row["model_name"]:
        ha = "right"
        va = "center"
        offset_y = 0
        offset_x = -5
    elif row["model_name"] == "effnet-b0":
        ha = "left"
        va = "center"
        offset_y = +0.0
        offset_x = 5
    elif row["model_name"] == "effnet-edgetpu-s":
        ha = "right"
    elif row["model_name"] == "mobilenetv3large":
        offset_x = -10

    plt.text(
        row["throughput_tpuv3"] + offset_x,
        row["float_acc"] + offset_y,
        row["model_name"],
        ha=ha,
        va=va,
        fontsize=fontsize,
    )

# ax.grid(True, linestyle='--', color='lightgray', alpha=0.7)
# Set axis labels and title
plt.subplots_adjust(left=0.30, right=0.85, bottom=0.13, top=0.9)
plt.xlabel("Thr. TPUv3 (images/sec/core)", fontsize=20)
plt.ylabel("Top-1 Accuracy (%)", fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=16)
plt.savefig("plot_results_tpuv3.pdf")
