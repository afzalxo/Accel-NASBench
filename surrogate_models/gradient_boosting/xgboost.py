import logging
import os
import pickle
import copy

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

import utils
from surrogate_model import SurrogateModel


class XGBModel(SurrogateModel):
    def __init__(
        self, data_root, log_dir, seed, model_config, data_config, device, metric
    ):
        super(XGBModel, self).__init__(
            data_root, log_dir, seed, model_config, data_config
        )
        self.model = None
        self.model_config["param:objective"] = "reg:squarederror"
        self.model_config["param:eval_metric"] = "rmse"

    def load_results_from_result_paths(self, result_paths):
        """
        Read in the result paths and extract hyperparameters and validation accuracy
        :param result_paths:
        :return:
        """
        # Get the train/test data
        hyps, val_accuracies = [], []
        for result_path in result_paths:
            config_space_instance, metrics, _ = self.config_loader[result_path]
            enc = config_space_instance.get_array()
            # print(enc)
            hyps.append(enc)
            val_accuracies.append(metrics[0])

        X = np.array(hyps)
        y = np.array(val_accuracies)

        # Impute none and nan values
        # Essential to prevent segmentation fault with robo
        idx = np.where(y is None)
        y[idx] = 100

        idx = np.isnan(X)
        X[idx] = -1

        return X, y

    def parse_param_config(self):
        identifier = "param:"
        param_config = dict()
        for key, val in self.model_config.items():
            if key.startswith(identifier):
                param_config[key.replace(identifier, "")] = val
        return param_config

    def train(self):
        X_train, y_train = self.load_results_from_result_paths(self.train_paths)
        X_val, y_val = self.load_results_from_result_paths(self.val_paths)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.model = xgb.train(
            param_config,
            dtrain,
            num_boost_round=self.model_config["param:num_rounds"],
            early_stopping_rounds=self.model_config["early_stopping_rounds"],
            verbose_eval=None,
            evals=[(dval, "val")],
        )

        train_pred, var_train = self.model.predict(dtrain), None
        val_pred, var_val = self.model.predict(dval), None

        # self.save()

        train_metrics = utils.evaluate_metrics(
            y_train, train_pred, prediction_is_first_arg=False
        )
        valid_metrics = utils.evaluate_metrics(
            y_val, val_pred, prediction_is_first_arg=False
        )

        logging.info("train metrics: %s", train_metrics)
        logging.info("valid metrics: %s", valid_metrics)

        self._metric = "Accuracy"
        self._metric_unit = "%"
        self._device = "-"

        fig_train = utils.scatter_plot(
            np.array(train_pred),
            np.array(y_train),
            xlabel=f"Predicted {self._metric} ({self._metric_unit})",
            ylabel=f"Measured {self._metric} ({self._metric_unit})",
            title=f"XGBoost fit on train set\n [{self._device}, {self._metric}]",
            metrics=copy.deepcopy(train_metrics),
        )
        fig_train.savefig(os.path.join(self.log_dir, "pred_vs_true_train.pdf"))
        plt.close()

        fig_val = utils.scatter_plot(
            np.array(val_pred),
            np.array(y_val),
            xlabel=f"Predicted {self._metric} ({self._metric_unit})",
            ylabel=f"Measured {self._metric} ({self._metric_unit})",
            title=f"XGBoost fit on val set\n [{self._device}, {self._metric}]",
            metrics=copy.deepcopy(valid_metrics),
        )
        fig_val.savefig(os.path.join(self.log_dir, "pred_vs_true_val.pdf"))
        plt.close()

        return valid_metrics

    def test(self):
        X_test, y_test = self.load_results_from_result_paths(self.test_paths)
        dtest = xgb.DMatrix(X_test, label=y_test)
        test_pred, var_test = self.model.predict(dtest), None

        test_metrics = utils.evaluate_metrics(
            y_test, test_pred, prediction_is_first_arg=False
        )

        logging.info("test metrics %s", test_metrics)

        fig = utils.scatter_plot(
            np.array(test_pred),
            np.array(y_test),
            xlabel=f"Predicted {self._metric} ({self._metric_unit})",
            ylabel=f"Measured {self._metric} ({self._metric_unit})",
            title=f"XGBoost fit on test set\n [{self._device}, {self._metric}]",
            metrics=copy.deepcopy(test_metrics),
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_test.pdf"))
        plt.close()
        return test_metrics

    def validate(self):
        X_val, y_val = self.load_results_from_result_paths(self.val_paths)
        dval = xgb.DMatrix(X_val, label=y_val)
        val_pred, var_val = self.model.predict(dval), None

        valid_metrics = utils.evaluate_metrics(
            y_val, val_pred, prediction_is_first_arg=False
        )

        logging.info("validation metrics %s", valid_metrics)

        return valid_metrics

    def save(self):
        pickle.dump(
            self.model, open(os.path.join(self.log_dir, "surrogate_model.model"), "wb")
        )

    def load(self, model_path):
        self.model = pickle.load(open(model_path, "rb"))

    def evaluate(self, result_paths):
        X_test, y_test = self.load_results_from_result_paths(result_paths)
        dtest = xgb.DMatrix(X_test, label=y_test)
        test_pred, var_test = self.model.predict(dtest), None

        test_metrics = utils.evaluate_metrics(
            y_test, test_pred, prediction_is_first_arg=False
        )
        return test_metrics, test_pred, y_test

    def query(self, configs):
        if isinstance(configs, list):
            pred = self.query_batch(configs)
        else:
            pred = self.query_config(configs)
        return pred

    def query_config(self, config_dict) -> float:
        config_space_instance = self.config_loader.query_config_dict(config_dict)
        X = config_space_instance.get_array().reshape(1, -1)
        idx = np.isnan(X)
        X[idx] = -1
        dtest = xgb.DMatrix(X)
        pred = self.model.predict(dtest)
        return pred

    def query_batch(self, config_batch):
        hyps = []
        for config_dict in config_batch:
            config_space_instance = self.config_loader.query_config_dict(config_dict)
            enc = config_space_instance.get_array()
            hyps.append(enc)

        X = np.array(hyps)
        dtest = xgb.DMatrix(X)
        pred = self.model.predict(dtest)
        return pred


class XGBModelAccel(XGBModel):
    def __init__(
        self, data_root, log_dir, seed, model_config, data_config, device, metric
    ):
        super(XGBModelAccel, self).__init__(
            data_root, log_dir, seed, model_config, data_config, device, metric
        )
        supported_dev = ["3090", "a100", "tpuv2", "tpuv3", "zcu102", "vck190"]
        supported_metrics = {
            "3090": ["throughput"],
            "a100": ["throughput"],
            "tpuv2": ["throughput"],
            "tpuv3": ["throughput"],
            "zcu102": ["latency", "throughput"],
            "vck190": ["latency", "throughput"],
        }
        assert (
            device.lower() in supported_dev
        ), f"Device {device} not supported... Select from {supported_dev}"
        assert (
            metric.lower() in supported_metrics[device.lower()]
        ), f"Performance metric {metric} not supported for device {device}"
        self.device = device
        self.metric = metric
        self.metric_unit = None
        if "tpu" in self.device:
            self.metric_unit = "images/sec/core"
        elif "zcu" or "vck" in self.device:
            if self.metric == "throughput":
                self.metric_unit = "images/sec"
            elif self.metric == "latency":
                self.metric_unit = "ms"
        elif self.device is not None:
            self.metric_unit = "images/sec"

        if self.device is None:
            self.metric = "Accuracy"
            self.metric_unit = "%"
            self.device = "-"

        self._surrogate = "XGBoost"

    # OVERRIDE
    def load_results_from_result_paths(self, result_paths):
        """
        Read in the result paths and extract hyperparameters and runtime
        :param result_paths:
        :return:
        """
        # Get the train/test data
        hyps, runtimes = [], []

        for result_path in result_paths:
            config_space_instance, runtime = self.config_loader.get_metric(
                result_path, self.device, self.metric
            )
            if runtime is None:
                continue
            hyps.append(config_space_instance.get_array())
            runtimes.append(float(runtime))

        X = np.array(hyps)
        y = np.array(runtimes)

        # Impute none and nan values
        # Essential to prevent segmentation fault with robo
        idx = np.where(y is None)
        y[idx] = 100

        idx = np.isnan(X)
        X[idx] = -1

        # return none to mimic return value of parent class
        return X, y

    def train(self):
        X_train, y_train = self.load_results_from_result_paths(self.train_paths)
        X_val, y_val = self.load_results_from_result_paths(self.val_paths)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        param_config = self.parse_param_config()
        param_config["seed"] = self.seed

        self.model = xgb.train(
            param_config,
            dtrain,
            num_boost_round=self.model_config["param:num_rounds"],
            early_stopping_rounds=self.model_config["early_stopping_rounds"],
            verbose_eval=None,
            evals=[(dval, "val")],
        )

        train_pred, var_train = self.model.predict(dtrain), None
        val_pred, var_val = self.model.predict(dval), None

        # self.save()

        fig_train = utils.scatter_plot(
            np.array(train_pred),
            np.array(y_train),
            xlabel=f"Predicted {self.metric} ({self.metric_unit})",
            ylabel=f"Measured {self.metric} ({self.metric_unit})",
            title=f"{self._surrogate} fit on train set\n [{self.device}, {self.metric}]",
        )
        fig_train.savefig(os.path.join(self.log_dir, "pred_vs_true_train.pdf"))
        plt.close()

        fig_val = utils.scatter_plot(
            np.array(val_pred),
            np.array(y_val),
            xlabel=f"Predicted {self.metric} ({self.metric_unit})",
            ylabel=f"Measured {self.metric} ({self.metric_unit})",
            title=f"{self._surrogate} fit on val set\n [{self.device}, {self.metric}]",
        )
        fig_val.savefig(os.path.join(self.log_dir, "pred_vs_true_val.pdf"))
        plt.close()

        train_metrics = utils.evaluate_metrics(
            y_train, train_pred, prediction_is_first_arg=False
        )
        valid_metrics = utils.evaluate_metrics(
            y_val, val_pred, prediction_is_first_arg=False
        )

        logging.info("train metrics: %s", train_metrics)
        logging.info("valid metrics: %s", valid_metrics)

        return valid_metrics

    def test(self):
        X_test, y_test = self.load_results_from_result_paths(self.test_paths)
        dtest = xgb.DMatrix(X_test, label=y_test)
        test_pred, var_test = self.model.predict(dtest), None

        test_metrics = utils.evaluate_metrics(
            y_test, test_pred, prediction_is_first_arg=False
        )

        fig = utils.scatter_plot(
            np.array(test_pred),
            np.array(y_test),
            xlabel=f"Predicted {self.metric} ({self.metric_unit})",
            ylabel=f"Measured {self.metric} ({self.metric_unit})",
            title=f"{self._surrogate} fit on test set\n [{self.device}, {self.metric}]",
            metrics=copy.deepcopy(test_metrics),
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_test.pdf"))
        plt.close()

        logging.info("test metrics %s", test_metrics)

        return test_metrics
