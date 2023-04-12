import logging
import os
import copy

import numpy as np
from sklearn.svm import NuSVR as sklearn_NuSVR
import matplotlib.pyplot as plt

import utils
from surrogate_models.svr.svr import SVR


class NuSVR(SVR):
    def __init__(self, data_root, log_dir, seed, model_config, data_config, device, metric):
        super(NuSVR, self).__init__(data_root, log_dir, seed, model_config, data_config, device, metric)
        # Instantiate model
        svr_config = {k:eval(v) for k,v in model_config.items() if v=='False' or
                      v=='True' and k != "device" and k != "metric"}
        self.model = sklearn_NuSVR(**svr_config)
        self._metric = 'Accuracy'
        self._metric_unit = '%'
        self._device = '-'
        self._surrogate = r'$\nu$-SVR'

    def train(self):
        X_train, y_train = self.load_results_from_result_paths(self.train_paths)
        X_val, y_val = self.load_results_from_result_paths(self.val_paths)
        self.model.fit(X_train, y_train)

        train_pred, var_train = self.model.predict(X_train), None
        val_pred, var_val = self.model.predict(X_val), None

        #self.save()
        train_metrics = utils.evaluate_metrics(y_train, train_pred, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_metrics(y_val, val_pred, prediction_is_first_arg=False)

        logging.info("train metrics: %s", train_metrics)
        logging.info("valid metrics: %s", valid_metrics)

        fig_train = utils.scatter_plot(
            np.array(train_pred),
            np.array(y_train),
            xlabel=f"Predicted {self._metric} ({self._metric_unit})",
            ylabel=f"Measured {self._metric} ({self._metric_unit})",
            title=f"{self._surrogate} fit on train set\n [{self._device}, {self._metric}]",
            metrics=copy.deepcopy(train_metrics)
        )
        fig_train.savefig(os.path.join(self.log_dir, "pred_vs_true_train.pdf"))
        plt.close()

        fig_val = utils.scatter_plot(
            np.array(val_pred),
            np.array(y_val),
            xlabel=f"Predicted {self._metric} ({self._metric_unit})",
            ylabel=f"Measured {self._metric} ({self._metric_unit})",
            title=f"{self._surrogate} fit on val set\n [{self._device}, {self._metric}]",
            metrics=copy.deepcopy(valid_metrics)
        )
        fig_val.savefig(os.path.join(self.log_dir, "pred_vs_true_val.pdf"))
        plt.close()

        return valid_metrics

    def test(self):
        X_test, y_test = self.load_results_from_result_paths(self.test_paths)
        test_pred, var_test = self.model.predict(X_test), None
        test_metrics = utils.evaluate_metrics(
            y_test, test_pred, prediction_is_first_arg=False
        )

        logging.info("test metrics %s", test_metrics)

        fig = utils.scatter_plot(
            np.array(test_pred),
            np.array(y_test),
            xlabel=f"Predicted {self._metric} ({self._metric_unit})",
            ylabel=f"Measured {self._metric} ({self._metric_unit})",
            title=f"{self._surrogate} fit on test set\n [{self._device}, {self._metric}]",
            metrics=copy.deepcopy(test_metrics)
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_test.pdf"))
        plt.close()
        return test_metrics


class NuSVRAccel(NuSVR):
    def __init__(self, data_root, log_dir, seed, model_config, data_config, device, metric):
        super(NuSVRAccel, self).__init__(data_root, log_dir, seed, model_config, data_config, device, metric)
        # Instantiate model
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
        if 'tpu' in self.device:
            self.metric_unit = 'images/sec/core'
        elif 'zcu' or 'vck' in self.device:
            if self.metric == 'throughput':
                self.metric_unit = 'images/sec'
            elif self.metric == 'latency':
                self.metric_unit = 'ms'
        elif self.device is not None:
            self.metric_unit = 'images/sec'

        self._metric = self.metric
        self._metric_unit = self.metric_unit
        self._device = self.device

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
            config_space_instance, runtime = self.config_loader.get_metric(result_path, self.device, self.metric)
            if runtime == None:
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
