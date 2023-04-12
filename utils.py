import glob
import json
import os
import re
from math import isclose

import ConfigSpace as CS
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ConfigSpace.read_and_write import json as config_space_json_r_w
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from surrogate_models.svr.svr import SVR
from surrogate_models.svr.svr import SVRAccel
from surrogate_models.svr.nu_svr import NuSVR
from surrogate_models.svr.nu_svr import NuSVRAccel
from surrogate_models.random_forest.sklearn_forest import SklearnForest
from surrogate_models.random_forest.sklearn_forest import SklearnForestAccel
from surrogate_models.gradient_boosting.xgboost import XGBModel
from surrogate_models.gradient_boosting.xgboost import XGBModelAccel
from surrogate_models.gradient_boosting.lgboost import LGBModel
from surrogate_models.gradient_boosting.lgboost import LGBModelAccel

sns.set_style('whitegrid')

model_dict = {
    'svr': SVR,
    'svr_accel': SVRAccel,
    'svr_nu': NuSVR,
    'svr_nu_accel': NuSVRAccel,
    'sklearn_forest': SklearnForest,
    'sklearn_forest_accel': SklearnForestAccel,
    'xgb': XGBModel,
    'lgb': LGBModel,
    'xgb_accel': XGBModelAccel,
    'lgb_accel': LGBModelAccel,
}


def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    # Adding MSE here
    metrics_dict['mae'] = mean_absolute_error(y_true, y_pred)
    metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
    metrics_dict["rmse"] = np.sqrt(metrics_dict["mse"])
    metrics_dict["r2"] = r2_score(y_true, y_pred)
    metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
    metrics_dict["kendall_tau_2_dec"], p_val = kendalltau(y_true, np.round(np.array(y_pred), decimals=2))
    metrics_dict["kendall_tau_1_dec"], p_val = kendalltau(y_true, np.round(np.array(y_pred), decimals=1))
    metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict


def get_model_configspace(model, device=None, metric=None):
    """
    Retrieve the model_config
    :param model: Name of the model for which you want the default config
    :return:
    """
    # Find matching config for the model name
    if device is not None and metric is not None:
        model_config_regex = re.compile(".*{}_{}_{}_configspace.json".format(model, device, metric))
    else:
        model_config_regex = re.compile(".*{}_configspace.json".format(model))
    matched_model_config_paths = list(
        filter(model_config_regex.match, glob.glob('configs/model_configs/*/*')))

    #print(matched_model_config_paths)
    # Make sure we only matched exactly one config
    assert len(matched_model_config_paths) == 1, 'Multiple or no configs matched with the requested model.'
    model_config_path = matched_model_config_paths[0]

    # Load the configspace object
    model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
    return model_configspace


def convert_array_to_list(a):
    """Converts a numpy array to list"""

    if isinstance(a, np.ndarray):
        return a.tolist()
    else:
        return a


def find_key_value(key, dictionary):
    """
    Check if key is contained in dictionary in a nested way
    Source: https://gist.github.com/douglasmiranda/5127251#file-gistfile1-py-L2
    :param key:
    :param dictionary:
    :return:
    """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find_key_value(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find_key_value(key, d):
                    yield result


def scatter_plot(xs, ys, xlabel, ylabel, title, metrics=None):
    """
    Creates scatter plot of the predicted and groundtruth performance
    :param xs:
    :param ys:
    :param xlabel:
    :param ylabel:
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(5, 4))
    plt.tight_layout()
    sns.set_style('darkgrid')
    sns.set_palette('deep')
    # plt.grid(True, which='both', ls='-', alpha=0.5)
    # plt.scatter(xs, ys, alpha=0.8, s=4)
    sns.scatterplot(x=xs, y=ys, marker='.', alpha=0.7, color='black', size=0.8)
    xs_min = xs.min()
    xs_max = xs.max()
    # plt.plot(np.linspace(xs_min, xs_max), np.linspace(xs_min, xs_max), 'r', alpha=0.5)
    sns.lineplot(x=np.linspace(xs_min, xs_max), y=np.linspace(xs_min, xs_max), alpha=0.5)
    # Get the axis object
    ax = plt.gca()

    # Set the xticks to have equal spacing
    xticks = ax.get_yticks()
    xticks = list(map(int, xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    yticks = xticks
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    # Add markers to the xticks and yticks
    ax.set_xticklabels(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks())
    if metrics is not None:
        metrics = pretty_metrics_dict(metrics)
        text_str = "\n".join([f"{k}$={v:.3f}$" for k, v in metrics.items()])
        plt.text(x=0.05, y=0.95, s=text_str, transform=plt.gca().transAxes, fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.legend().remove()
    plt.title(title)
    plt.axis('square')
    return fig


def pretty_metrics_dict(metrics):
    if 'mae' in metrics.keys():
        mae = metrics.pop('mae')
        metrics['MAE'] = mae
    if 'rmse' in metrics.keys():
        mae = metrics.pop('rmse')
        metrics['RMSE'] = mae
    if 'kendall_tau' in metrics.keys():
        kt = metrics['kendall_tau']
        del metrics['kendall_tau']
        metrics[r"Kendall's Tau $\tau$"] = kt
    if 'kendall_tau_2_dec' in metrics.keys():
        del metrics['kendall_tau_2_dec']
    if 'kendall_tau_1_dec' in metrics.keys():
        del metrics['kendall_tau_1_dec']
    if 'mse' in metrics.keys():
        del metrics['mse']
    if 'spearmanr' in metrics.keys():
        spearmanr = metrics['spearmanr']
        del metrics['spearmanr']
        metrics[r"Spearman's Rank $\rho$"] = spearmanr
    if 'r2' in metrics.keys():
        kt = metrics['r2']
        del metrics['r2']
        metrics[r'$R^2$'] = kt
    return metrics


def plot_predictions(mu_train, mu_test, var_train, var_test, train_y, test_y,
                     log_dir, name='random forest', x1=0, x2=100, y1=0, y2=100):
    f, ax = plt.subplots(1, 2, figsize=(15, 6))

    if var_train is not None:
        ll = norm.logpdf(np.array(train_y, dtype=np.float), loc=mu_train, scale=np.sqrt(var_train))
        c_map = 'viridis'
    else:
        ll = 'b'
        c_map = None

    im1 = ax[0].scatter(mu_train, train_y, c=ll, cmap=c_map)
    ax[0].set_xlabel('predicted', fontsize=15)
    ax[0].set_ylabel('true', fontsize=15)
    ax[0].set_title('{} (train)'.format(name), fontsize=15)
    ax[0].plot([0, 100], [0, 100], 'k--')
    if var_train is not None:
        f.colorbar(im1, ax=ax[0])

    if var_test is not None:
        ll = norm.logpdf(np.array(test_y, dtype=np.float), loc=mu_test, scale=np.sqrt(var_test))
        c_map = 'viridis'
    else:
        ll = 'b'
        c_map = None

    ax[1].set_xlim([x1, x2])
    ax[1].set_ylim([y1, y2])

    im1 = ax[1].scatter(mu_test, test_y, c=ll, cmap=c_map)
    ax[1].set_xlabel('predicted', fontsize=15)
    ax[1].set_ylabel('true', fontsize=15)
    ax[1].set_title('{} (test)'.format(name), fontsize=15)
    ax[1].plot([0, 100], [0, 100], 'k--')
    if var_test is not None:
        f.colorbar(im1, ax=ax[1])
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, '_'.join(name.split()) + '.jpg'))
    return plt.gcf()


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ConfigLoader:
    def __init__(self, config_space_path):
        self.config_space = self.load_config_space(config_space_path)

        # Manually adjust a certain set of hyperparameters
        self.parameter_change_dict = None

    def __getitem__(self, path):
        """
        Load the results from results.json
        :param path: Path to results.json
        :return:
        """
        json_file = json.load(open(path, 'r'))
        config_dict = json_file['architecture']

        config_space_instance = self.query_config_dict(config_dict)
        val_top1 = json_file['metrics']['val_top1']
        val_top5 = json_file['metrics']['val_top1']
        train_time = json_file['metrics']['train_time']
        return config_space_instance, [val_top1, val_top5, train_time], json_file

    def get_metric(self, path, device, metric):
        """
        Load a performance metric from results.json
        :param path: Path to results.json
        return:
        """
        json_file = json.load(open(path, 'r'))
        config_dict = json_file['architecture']

        config_space_instance = self.query_config_dict(config_dict)
        metric_val = json_file['platform_perf'][f'{metric}_{device}'][f'{metric}_mean']
        return config_space_instance, metric_val

    def query_config_dict(self, config_dict):
        # Change a selection of parameters
        if self.parameter_change_dict is not None:
            config_dict = self.change_parameter(config_dict)

        # Create the config space instance based on the config space
        config_space_instance = \
            self.convert_config_dict_to_configspace_instance(self.config_space, config_dict=config_dict)

        return config_space_instance

    def change_parameter(self, config_dict):
        for name, value in self.parameter_change_dict.items():
            config_dict[name] = value
        return config_dict

    def convert_config_dict_to_configspace_instance(self, config_space, config_dict):
        """
        Convert a config dictionary to configspace instace
        :param config_space:
        :param config_dict:
        :return:
        """

        def _replace_str_bool_with_python_bool(input_dict):
            for key, value in input_dict.items():
                if value == 'True':
                    input_dict[key] = True
                elif value == 'False':
                    input_dict[key] = False
                else:
                    pass
            return input_dict

        # Replace the str true with python boolean type
        config_dict = _replace_str_bool_with_python_bool(config_dict)
        config_instance = CS.Configuration(config_space, values=config_dict)
        return config_instance

    @staticmethod
    def load_config_space(path):
        """
        Load ConfigSpace object
        As certain hyperparameters are not denoted as optimizable but overriden later,
        they are manually overriden here too.
        :param path:
        :return:
        """
        with open(os.path.join(path), 'r') as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)

        return config_space


class ResultLoader:
    def __init__(self, root, filepath_regex, train_val_test_split, seed):
        self.root = root
        self.filepath_regex = filepath_regex
        self.train_val_test_split = train_val_test_split
        np.random.seed(seed)

    def return_train_val_test(self):  #TODO: REMOVE portion
        """
        Get the result train/val/test split.
        :return:
        """
        # TODO: REMOVE NEXT 2 LINE
        # self.train_val_test_split['type'] = 'portion_result_paths'
        # self.train_val_test_split['portion'] = portion

        if self.train_val_test_split['type'] == 'all_result_paths':
            paths_split = self.all_result_paths()
        elif self.train_val_test_split['type'] == 'portion_result_paths':
            paths_split = self.portion_result_paths()
        elif self.train_val_test_split['type'] == 'no_data':
            paths_split = [], [], []
        else:
            raise ValueError('Unknown train/val/test split.')
        train_paths, val_paths, test_paths = paths_split
        return train_paths, val_paths, test_paths

    def filter_duplicate_dirs(self, paths_to_json):
        """
        Checks to configurations in the results.json files and returns paths such that none contains
        duplicate configurations.
        :param paths_to_json: List of dir/results.json
        :return: unique list of dir/results.json w.r.t. configuration
        """
        config_hashes = []

        for path_to_json in paths_to_json:
            with open(path_to_json, "r") as f:
                results = json.load(f)
            config_hash = hash(results["architecture"].__repr__())
            config_hashes.append(config_hash)

        _, unique_indices = np.unique(config_hashes, return_index=True)

        return list(np.array(paths_to_json)[unique_indices])

    def get_splits(self, paths, ratios=None):
        """
        Divide the paths into train/val/test splits.
        :param paths:
        :param ratios:
        :return:
        """
        if ratios is None:
            train_ratio, val_ratio, test_ratio = self.train_val_test_split['train'], self.train_val_test_split['val'], \
                                                 self.train_val_test_split['test']
        else:
            train_ratio, val_ratio, test_ratio = ratios
        assert isclose(train_ratio + val_ratio + test_ratio, 1.0,
                       abs_tol=1e-8), 'The train/val/test split should add up to 1.'

        # Randomly shuffle the list
        rng = np.random.RandomState(6)
        rng.shuffle(paths)

        # Extract the train/val/test splits
        train_upper_idx = int(train_ratio * len(paths))
        val_upper_idx = int((train_ratio + val_ratio) * len(paths))

        train_paths = paths[:train_upper_idx]
        val_paths = paths[train_upper_idx:val_upper_idx]
        test_paths = paths[val_upper_idx:-1]
        return train_paths, val_paths, test_paths

    def all_result_paths(self):
        """
        Return the paths of all results
        :return: result paths
        """
        all_results_paths = glob.glob(os.path.join(self.root, self.filepath_regex))
        print("==> Found %i results paths. Filtering duplicates..." % len(all_results_paths))
        all_results_paths.sort()
        all_results_paths_filtered = self.filter_duplicate_dirs(all_results_paths)
        print("==> Finished filtering. Found %i unique architectures, %i duplicates" % (len(all_results_paths_filtered), \
                                                                                        len(all_results_paths) - len(
                                                                                            all_results_paths_filtered)))
        train_paths, val_paths, test_paths = self.get_splits(all_results_paths_filtered)
        return train_paths, val_paths, test_paths

    def portion_result_paths(self):
        portion = self.train_val_test_split['portion']
        all_results_paths = glob.glob(os.path.join(self.root, self.filepath_regex))
        print("==> Found %i results paths. Filtering duplicates..." % len(all_results_paths))
        all_results_paths_filtered = self.filter_duplicate_dirs(all_results_paths)
        print("==> Finished filtering. Found %i unique architectures, %i duplicates" % (len(all_results_paths_filtered), \
                                                                                        len(all_results_paths) - len(
                                                                                            all_results_paths_filtered)))
        rng = np.random.RandomState(6)
        rng.shuffle(all_results_paths_filtered)
        if portion < 1:
            portion = len(all_results_paths_filtered) * portion
        all_results_paths_filtered = all_results_paths_filtered[:portion]
        all_results_paths_filtered.sort()
        train_paths, val_paths, test_paths = self.get_splits(all_results_paths_filtered)
        return train_paths, val_paths, test_paths
