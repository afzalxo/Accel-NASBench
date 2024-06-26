import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

#from nasbench301.surrogate_models import utils
#from nasbench301.surrogate_models.bananas.bananas_utils import BANANASDataset
#from nasbench301.surrogate_models.surrogate_model import SurrogateModel
import utils
from surrogate_model import SurrogateModel


class NGBModel(SurrogateModel):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        super(NGBModel, self).__init__(data_root, log_dir, seed, model_config, data_config)
        self.model = None

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
            #print(enc)
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

    def parse_config(self, identifier):
        param_config = dict()
        for key, val in self.model_config.items():
            if key.startswith(identifier):
                param_config[key.replace(identifier, "")] = val
        return param_config

    def train(self):
        X_train, y_train = self.load_results_from_result_paths(self.train_paths)
        X_val, y_val = self.load_results_from_result_paths(self.val_paths)

        base_learner_config = self.parse_config("base:")
        param_config = self.parse_config("param:")

        # train
        base_learner = DecisionTreeRegressor(criterion='friedman_mse', random_state=None, splitter='best',
                                             **base_learner_config)
        self.model = NGBRegressor(Dist=Normal, Base=base_learner, Score=LogScore, verbose=True, **param_config)
        self.model = self.model.fit(X_train, y_train, X_val=X_val, Y_val=y_val,
                                    early_stopping_rounds=self.model_config["early_stopping_rounds"])

        train_pred, var_train = self.model.predict(X_train), None
        val_pred, var_val = self.model.predict(X_val), None

        # self.save()

        fig_train = utils.scatter_plot(np.array(train_pred), np.array(y_train), xlabel='Predicted', ylabel='True',
                                       title='')
        fig_train.savefig(os.path.join(self.log_dir, 'pred_vs_true_train.jpg'))
        plt.close()

        fig_val = utils.scatter_plot(np.array(val_pred), np.array(y_val), xlabel='Predicted', ylabel='True', title='')
        fig_val.savefig(os.path.join(self.log_dir, 'pred_vs_true_val.jpg'))
        plt.close()

        train_metrics = utils.evaluate_metrics(y_train, train_pred, prediction_is_first_arg=False)
        valid_metrics = utils.evaluate_metrics(y_val, val_pred, prediction_is_first_arg=False)

        logging.info('train metrics: %s', train_metrics)
        logging.info('valid metrics: %s', valid_metrics)

        return valid_metrics

    def test(self):
        X_test, y_test = self.load_results_from_result_paths(self.test_paths)
        test_pred, var_test = self.model.predict(X_test), None

        fig = utils.scatter_plot(np.array(test_pred), np.array(y_test), xlabel='Predicted', ylabel='True', title='')
        fig.savefig(os.path.join(self.log_dir, 'pred_vs_true_test.jpg'))
        plt.close()

        test_metrics = utils.evaluate_metrics(y_test, test_pred, prediction_is_first_arg=False)

        logging.info('test metrics %s', test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val = self.load_results_from_result_paths(self.val_paths)
        val_pred, var_val = self.model.predict(X_val), None

        valid_metrics = utils.evaluate_metrics(y_val, val_pred, prediction_is_first_arg=False)

        logging.info('validation metrics %s', valid_metrics)

        return valid_metrics

    def save(self):
        pickle.dump(self.model, open(os.path.join(self.log_dir, 'surrogate_model.model'), 'wb'))

    def load(self, model_path):
        self.model = pickle.load(open(model_path, 'rb'))

    def evaluate(self, result_paths):
        X_test, y_test = self.load_results_from_result_paths(result_paths)
        test_pred, var_test = self.model.predict(X_test), None

        test_metrics = utils.evaluate_metrics(y_test, test_pred, prediction_is_first_arg=False)
        return test_metrics, test_pred, y_test

    def query(self, config_dict):
        config_space_instance = self.config_loader.query_config_dict(config_dict)
        X = config_space_instance.get_array().reshape(1, -1)
        idx = np.isnan(X)
        X[idx] = -1
        pred = self.model.predict(X)
        return pred
