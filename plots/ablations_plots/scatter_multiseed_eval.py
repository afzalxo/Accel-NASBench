import sys
import numpy as np
import statistics
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append("./")
import accelnb as anb
from configurationspaces.searchspaces import EfficientNetSS as ss


def obtain_predicted_acc_surrogate(dataframe, surrogate, subdir=None):
    ensemble_inst_acc = anb.ANBEnsemble(surrogate, seed=3, subdir=subdir)
    acc_model = ensemble_inst_acc.load_ensemble()
    search_space = ss()

    exps = dataframe.iloc[:, 9:16].values.tolist()
    kernels = dataframe.iloc[:, 16:23].values.tolist()
    layers = dataframe.iloc[:, 23:30].values.tolist()
    se = dataframe.iloc[:, 30:37].values.tolist()

    test_samples = []
    for exp, kern, lay, s in zip(exps, kernels, layers, se):
        test_samples.append(search_space.manual_sample([exp, kern, lay, s]))
    mean_acc, std = acc_model.query(test_samples)
    return mean_acc, std


def pretty_metrics_dict(metrics):
    if "kl" in metrics.keys():
        kl = metrics.pop("kl")
        metrics["KL div."] = kl
    if "mae" in metrics.keys():
        mae = metrics.pop("mae")
        metrics["MAE"] = mae
    if "rmse" in metrics.keys():
        mae = metrics.pop("rmse")
        metrics["RMSE"] = mae
    if "kendall_tau" in metrics.keys():
        kt = metrics["kendall_tau"]
        del metrics["kendall_tau"]
        metrics[r"Kendall's Tau $\tau$"] = kt
    if "kendall_tau_2_dec" in metrics.keys():
        del metrics["kendall_tau_2_dec"]
    if "kendall_tau_1_dec" in metrics.keys():
        del metrics["kendall_tau_1_dec"]
    if "mse" in metrics.keys():
        del metrics["mse"]
    if "spearmanr" in metrics.keys():
        spearmanr = metrics["spearmanr"]
        del metrics["spearmanr"]
        metrics[r"Spearman's Rank $\rho$"] = spearmanr
    if "r2" in metrics.keys():
        kt = metrics["r2"]
        del metrics["r2"]
        metrics[r"$R^2$"] = kt
    return metrics


def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    metrics_dict["kl"] = entropy(y_true, y_pred)
    metrics_dict["mae"] = mean_absolute_error(y_true, y_pred)
    metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
    metrics_dict["rmse"] = np.sqrt(metrics_dict["mse"])
    metrics_dict["r2"] = r2_score(y_true, y_pred)
    metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
    metrics_dict["kendall_tau_2_dec"], p_val = kendalltau(
        y_true, np.round(np.array(y_pred), decimals=2)
    )
    metrics_dict["kendall_tau_1_dec"], p_val = kendalltau(
        y_true, np.round(np.array(y_pred), decimals=1)
    )

    metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict


df_p_seed1 = pd.read_csv("plots/ablations_plots/csvs/proxified-seed1.csv", header=None)
predicted_accs_xgb, _ = obtain_predicted_acc_surrogate(
    df_p_seed1, surrogate="xgb", subdir=None
)
predicted_accs_lgb, _ = obtain_predicted_acc_surrogate(
    df_p_seed1, surrogate="lgb", subdir=None
)

df_p_seed1 = df_p_seed1.drop(df_p_seed1.columns[list(range(9, 9 + 28))], axis=1)
df_p_seed1.columns = [
    "Junk0",
    "JID",
    "Model Num",
    "Model Seed",
    "Acc Top-1",
    "Acc Top-5",
    "FLOPs",
    "MParams",
    "Train Time",
]
df_p_seed1.FLOPs = df_p_seed1.FLOPs.round()
df_p_seed1.FLOPs = pd.to_numeric(df_p_seed1.FLOPs, downcast="integer")
del df_p_seed1["Junk0"]
p_tt_s1 = df_p_seed1["Train Time"].sum() / (60 * 60)

df_p_seed2 = pd.read_csv("plots/ablations_plots/csvs/proxified-seed2.csv", header=None)
df_p_seed2 = df_p_seed2.drop(df_p_seed2.columns[list(range(9, 9 + 28))], axis=1)
df_p_seed2.columns = [
    "Junk0",
    "JID",
    "Model Num",
    "Model Seed",
    "Acc Top-1",
    "Acc Top-5",
    "FLOPs",
    "MParams",
    "Train Time",
]
del df_p_seed2["Junk0"]
p_tt_s2 = df_p_seed2["Train Time"].sum() / (60 * 60)

df_p_seed3 = pd.read_csv("plots/ablations_plots/csvs/proxified-seed3.csv", header=None)
df_p_seed3 = df_p_seed3.drop(df_p_seed3.columns[list(range(9, 9 + 28))], axis=1)
df_p_seed3.columns = [
    "Junk0",
    "JID",
    "Model Num",
    "Model Seed",
    "Acc Top-1",
    "Acc Top-5",
    "FLOPs",
    "MParams",
    "Train Time",
]
del df_p_seed3["Junk0"]
p_tt_s3 = df_p_seed3["Train Time"].sum() / (60 * 60)

df_fp_seed1 = pd.read_csv("plots/ablations_plots/csvs/fewproxy-seed1.csv", header=None)
df_fp_seed1 = df_fp_seed1.drop(df_fp_seed1.columns[list(range(9, 9 + 28))], axis=1)
df_fp_seed1.columns = [
    "Junk0",
    "JID",
    "Model Num",
    "Model Seed",
    "Acc Top-1",
    "Acc Top-5",
    "FLOPs",
    "MParams",
    "Train Time",
]
del df_fp_seed1["Junk0"]
fp_tt_s1 = df_fp_seed1["Train Time"].sum() / (60 * 60)

df_fp_seed2 = pd.read_csv("plots/ablations_plots/csvs/fewproxy-seed2.csv", header=None)
df_fp_seed2 = df_fp_seed2.drop(df_fp_seed2.columns[list(range(9, 9 + 28))], axis=1)
df_fp_seed2.columns = [
    "Junk0",
    "JID",
    "Model Num",
    "Model Seed",
    "Acc Top-1",
    "Acc Top-5",
    "FLOPs",
    "MParams",
    "Train Time",
]
del df_fp_seed2["Junk0"]
fp_tt_s2 = df_fp_seed2["Train Time"].sum() / (60 * 60)

df_fp_seed3 = pd.read_csv("plots/ablations_plots/csvs/fewproxy-seed3.csv", header=None)
df_fp_seed3 = df_fp_seed3.drop(df_fp_seed3.columns[list(range(9, 9 + 28))], axis=1)
df_fp_seed3.columns = [
    "Junk0",
    "JID",
    "Model Num",
    "Model Seed",
    "Acc Top-1",
    "Acc Top-5",
    "FLOPs",
    "MParams",
    "Train Time",
]
del df_fp_seed3["Junk0"]
fp_tt_s3 = df_fp_seed3["Train Time"].sum() / (60 * 60)

hours_to_train_fp = fp_tt_s1 + fp_tt_s2 + fp_tt_s3
hours_to_train_p = p_tt_s1 + p_tt_s2 + p_tt_s3
print(
    "Hours to Train using FP and P training schemes (Using a single node with 4 GPUs):"
)
print(hours_to_train_fp, hours_to_train_p)
node_wattage = 1.5  # Node consumes roughly 1.5kW during training
carbon_intensity = 0.71  # Local carbon intesity in kg CO2-e/kWH (For the year 2021) (TODO: Source redacted for peer-review)
co2_emissions_p = (
    node_wattage * carbon_intensity * hours_to_train_p
)  # Amount of CO2 emitted in kg
co2_emissions_fp = node_wattage * carbon_intensity * hours_to_train_fp
print("CO_2 emissions from FP and P training schemes:")
print(co2_emissions_fp, co2_emissions_p)

seed1_pred_prox_xgb = predicted_accs_xgb
seed1_pred_prox_lgb = predicted_accs_lgb
seed1_prox = list(df_p_seed1["Acc Top-1"])
seed2_prox = list(df_p_seed2["Acc Top-1"])
seed3_prox = list(df_p_seed3["Acc Top-1"])
seed1_fewprox = list(df_fp_seed1["Acc Top-1"])
seed2_fewprox = list(df_fp_seed2["Acc Top-1"])
seed3_fewprox = list(df_fp_seed3["Acc Top-1"])

mean_prox = [statistics.mean(x) for x in zip(seed1_prox, seed2_prox, seed3_prox)]
mean_fewprox = [
    statistics.mean(x) for x in zip(seed1_fewprox, seed2_fewprox, seed3_fewprox)
]
std_prox = [statistics.stdev(x) for x in zip(seed1_prox, seed2_prox, seed3_prox)]
std_fewprox = [
    statistics.stdev(x) for x in zip(seed1_fewprox, seed2_fewprox, seed3_fewprox)
]

df_tabular = pd.DataFrame(
    {
        "mean_fewprox": mean_fewprox,
        "mean_prox": mean_prox,
        "std_fewprox": std_fewprox,
        "std_prox": std_prox,
    }
)

mean_prox_mean_fewprox = evaluate_metrics(mean_prox, mean_fewprox, False)

sns.set_style("darkgrid")
sns.set_palette("deep")
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, right=0.70, bottom=0.2, top=0.9)
sns.scatterplot(
    data=df_tabular, x="mean_fewprox", y="mean_prox", marker=".", ax=ax, color="black"
)
ax.errorbar(
    mean_fewprox,
    mean_prox,
    xerr=std_fewprox,
    yerr=std_prox,
    ls="none",
    linewidth=0.5,
    color="gray",
)

palette = sns.color_palette("tab10")
sns.set_style(style="whitegrid")
width = 0.8
co2_emissions = [co2_emissions_p, co2_emissions_fp]
categories = ["Proxified", "Few-Proxy"]
barh_ax = ax.inset_axes([0.65, 0.15, 0.3, 0.18])
barh_ax.grid(axis="y", linewidth=0)
barh_ax.barh([1, 2], co2_emissions, height=width, color=palette[2:4])
arrow_length = 0.5
barh_ax.annotate(
    "",
    xy=(co2_emissions[1], 1),
    xytext=(co2_emissions[0], 1),
    arrowprops=dict(
        arrowstyle="<->", lw=1.0, mutation_scale=10, color="black", shrinkA=arrow_length
    ),
)
barh_ax.text(
    (co2_emissions[0] + co2_emissions[1]) / 2,
    1 - 0.5,
    "~5.6x",
    ha="center",
    va="center",
    color="red",
)

barh_ax.set_ylim((0, 3))
barh_ax.set_xlim((0, max(co2_emissions) * 1.1))
barh_ax.set_yticks([i + width / 2 for i in [0.6, 1.8]])
barh_ax.set_yticklabels(categories, fontsize=10)
barh_ax.set_xlabel("CO$_2$ emissions (Kg)", fontsize=10)
barh_ax.set_title("Training CO$_2$ Cost", fontsize=10)
ax.set_xlabel("Top-1 Acc, Few-Proxy", fontsize=18)
ax.set_ylabel("Top-1 Acc, Proxified", fontsize=18)
ax.tick_params(axis="x", labelsize=16, labelcolor="red")
ax.tick_params(axis="y", labelsize=16, labelcolor="green")

metrics = pretty_metrics_dict(mean_prox_mean_fewprox)
del metrics[r"$R^2$"]
del metrics["MAE"]
del metrics["RMSE"]
kl_div = metrics["KL div."]
text_str = f"KL div.$={kl_div:.2e}$\n"
del metrics["KL div."]
text_str = text_str + "\n".join([f"{k}$={v:.3f}$" for k, v in metrics.items()])
text_str += f"\nMean $\\sigma$ Proxified$={statistics.mean(std_prox)/100:.2e}$"
text_str += f"\nMean $\\sigma$ Few-Proxy$={statistics.mean(std_fewprox)/100:.2e}$"
plt.text(
    x=0.05,
    y=0.95,
    s=text_str,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.savefig("underfitting_vs_true.pdf")

gtruth_prox = list((np.array(seed2_prox) + np.array(seed3_prox)) / 2)
gtruth_fewprox = list((np.array(seed2_fewprox) + np.array(seed3_fewprox)) / 2)

prox_pred_xgb_metrics = evaluate_metrics(seed1_pred_prox_xgb, gtruth_prox, False)
prox_pred_lgb_metrics = evaluate_metrics(seed1_pred_prox_lgb, gtruth_prox, False)
fewprox_pred_xgb_metrics = evaluate_metrics(seed1_pred_prox_xgb, gtruth_fewprox, False)
fewprox_pred_lgb_metrics = evaluate_metrics(seed1_pred_prox_lgb, gtruth_fewprox, False)
tab_fewprox_true_fewprox = evaluate_metrics(seed1_fewprox, gtruth_fewprox, False)
tab_prox_true_fewprox = evaluate_metrics(seed2_prox, gtruth_fewprox, False)
tab_prox_true_prox = evaluate_metrics(seed1_prox, gtruth_prox, False)
print("Proxified Pred vs Proxified GT (XGB):")
print(prox_pred_xgb_metrics)
print("Proxified Pred vs Proxified GT (LGB):")
print(prox_pred_lgb_metrics)
print("Proxified Pred vs FewProx GT (XGB):")
print(fewprox_pred_xgb_metrics)
print("Proxified Pred vs FewProx GT (LGB):")
print(fewprox_pred_lgb_metrics)
print("Tabular FewProx vs True FewProx GT:")
print(tab_fewprox_true_fewprox)
print("Tabular Proxified vs True FewProx GT:")
print(tab_prox_true_fewprox)
print("Tabular Proxified vs True Proxified GT:")
print(tab_prox_true_prox)
print("===" * 10)

# Remove this section
gtruth_prox13 = list((np.array(seed1_prox) + np.array(seed2_prox)) / 2)
tab_prox_true_prox = evaluate_metrics(seed3_prox, gtruth_prox13, False)
prox_pred_xgb_metrics = evaluate_metrics(seed1_pred_prox_xgb, gtruth_prox13, False)
prox_pred_lgb_metrics = evaluate_metrics(seed1_pred_prox_lgb, gtruth_prox13, False)
print("Tabular Prox 2 vs True Proxified GT 13:")
print(tab_prox_true_prox)
print("Pred vs True Proxified GT 13:")
print(prox_pred_xgb_metrics)
print("Pred vs True Proxified GT 13:")
print(prox_pred_lgb_metrics)
###

print("----" * 8)
maes_proxified_avgs = []
maes_proxified_stds = []
kt_proxified = []
kt_proxified_stds = []
kt_fewproxies = []
kt_fewprox_stds = []

sr_proxified = []
kl_proxified = []
sr_fewprox = []
kl_fewproxies = []
num_samples = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
df = pd.read_csv("plots/ablations_plots/csvs/proxified-seed2.csv", header=None)

for q in num_samples:
    maes_prox = []
    kt_prox = []
    kt_fewprox = []
    kl_fewprox = []
    for seed in range(1, 6):
        subdir = f"experiments/ablation_models/xgb/model_{q}samples-{seed}"
        predicted_accs_xgb, _ = obtain_predicted_acc_surrogate(
            df, surrogate="xgb", subdir=subdir
        )
        prox_pred_xgb_metrics = evaluate_metrics(predicted_accs_xgb, gtruth_prox, False)
        fewprox_pred_xgb_metrics = evaluate_metrics(
            predicted_accs_xgb, gtruth_fewprox, False
        )
        # print(f'Proxified Pred Size: {q} vs Proxified GT (XGB):')
        # print(prox_pred_xgb_metrics)
        # print(f'Proxified Pred Size: {q} vs FewProx GT (XGB):')
        # print(fewprox_pred_xgb_metrics)
        # print('----'*8)
        maes_prox.append(prox_pred_xgb_metrics["mae"])
        kt_prox.append(prox_pred_xgb_metrics["kendall_tau"])
        kt_fewprox.append(fewprox_pred_xgb_metrics["kendall_tau"])
        kl_fewprox.append(fewprox_pred_xgb_metrics["kl"])

    maes_proxified_avgs.append(np.mean(maes_prox))
    maes_proxified_stds.append(np.std(maes_prox))
    kt_proxified.append(np.mean(kt_prox))
    kt_proxified_stds.append(np.std(kt_prox))

    kt_fewproxies.append(np.mean(kt_fewprox))
    kt_fewprox_stds.append(np.std(kt_fewprox))
    kl_fewproxies.append(np.mean(kl_fewprox))

maes_proxified_avgs = list(np.array(maes_proxified_avgs) / 100)

# Following code mostly generated using ChatGPT

# Set style and color palette
sns.set_style("darkgrid")
sns.set_palette("deep")

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(8, 7))
ax2 = ax1.twinx()
palette = sns.color_palette("tab10")
# Plot the first line (maes_proxified)
sns.lineplot(
    x=num_samples,
    y=maes_proxified_avgs,
    color=palette[1],
    markers=True,
    marker="o",
    markersize=14,
    linewidth=3,
    ax=ax2,
)
sns.lineplot(
    x=num_samples,
    y=list(np.ones_like(num_samples) * tab_prox_true_prox["mae"] / 100),
    color=palette[1],
    linestyle="--",
    linewidth=3.0,
    markers=False,
    ax=ax2,
)

# Plot the second line (kt_proxified)
sns.lineplot(
    x=num_samples,
    y=kt_proxified,
    color=palette[0],
    markers=True,
    marker="*",
    markersize=20,
    linewidth=3,
    ax=ax1,
)
sns.lineplot(
    x=num_samples,
    y=list(np.ones_like(num_samples) * tab_prox_true_prox["kendall_tau"]),
    color=palette[0],
    linestyle="--",
    linewidth=3.0,
    markers=False,
    ax=ax1,
)

# Set axis labels and title
ax1.set_xlabel("Training data size", fontsize=22)
ax2.set_ylabel("Mean Absolute Error", fontsize=22)
ax1.set_ylabel("Kendall's Tau $\\tau$", fontsize=22)

# Show the plot
plt.savefig("training_size-vs-mae-kt_proxified.pdf")
plt.clf()

fig, ax1 = plt.subplots(figsize=(8, 7))
ax2 = ax1.twinx()
plt.subplots_adjust(left=0.15, right=0.80, bottom=0.2, top=0.9)

# Plot the second line (kt_proxified vs fewproxy)
sns.lineplot(
    x=num_samples,
    y=kt_fewproxies,
    color=palette[0],
    markers=True,
    marker="*",
    markersize=20,
    linewidth=3,
    ax=ax1,
)
sns.lineplot(
    x=num_samples,
    y=kl_fewproxies,
    color=palette[1],
    markers=True,
    marker="o",
    markersize=12,
    linewidth=3,
    ax=ax2,
)
sns.lineplot(
    x=num_samples,
    y=list(np.ones_like(num_samples) * tab_prox_true_fewprox["kendall_tau"]),
    color=palette[0],
    linestyle="--",
    linewidth=3.0,
    markers=False,
    ax=ax1,
)
sns.lineplot(
    x=num_samples,
    y=list(np.ones_like(num_samples) * tab_prox_true_fewprox["kl"]),
    color=palette[1],
    linestyle="--",
    linewidth=3.0,
    markers=False,
    ax=ax2,
)

# Set axis labels and title
ax1.set_xlabel("Training data size", fontsize=22)
ax1.set_ylabel("Kendall's Tau $\\tau$", fontsize=22)
ax2.set_ylabel("KL div. (1e-4)", fontsize=22)
ax1.tick_params(axis="y", labelsize=20, labelcolor=palette[0])
ax1.tick_params(axis="x", labelsize=20)
ax2.tick_params(axis="y", labelsize=20, labelcolor=palette[1])
ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax2.yaxis.get_offset_text().set_fontsize(0)
legend_entries = [
    plt.Line2D(
        [0],
        [0],
        marker="*",
        markersize=10,
        color=palette[0],
        label="Surrogate KT $\\tau$",
        linestyle="-",
    ),
    plt.Line2D(
        [0],
        [0],
        markersize=20,
        color=palette[0],
        label="Tabular Proxified KT $\\tau$",
        linestyle="--",
    ),
    plt.Line2D(
        [0],
        [0],
        marker="o",
        markersize=8,
        color=palette[1],
        label="Surrogate KL div.",
        linestyle="-",
    ),
    plt.Line2D(
        [0],
        [0],
        markersize=20,
        color=palette[1],
        label="Tabular Proxified KL div.",
        linestyle="--",
    ),
]

plt.legend(handles=legend_entries, loc="center right", fontsize=16)

# Show the plot
plt.savefig("training_size-vs-kt_fewproxy.pdf")
