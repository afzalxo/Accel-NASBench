import json
import os
import time
import argparse

import matplotlib
import numpy as np
from sklearn.model_selection import KFold

from ConfigSpace.read_and_write import json as config_space_json_r_w
import utils

from ensemble import Ensemble

matplotlib.use("Agg")


def train_surrogate_model():

    parser = argparse.ArgumentParser("Surrogate Train")
    parser.add_argument("--dataset_root", type=str, help="Path to dataset root dir")
    parser.add_argument("--model", type=str, help="Surrogate model")
    parser.add_argument("--model_config_path", type=str, default=None, help="Model config path")
    parser.add_argument("--data_config_path", type=str, default=None, help="Data config path")
    parser.add_argument("--device", default=None, type=str, help="If using performance model, which device?")
    parser.add_argument("--metric", default=None, type=str, help="If using performance model for a device, which metric to fit?")
    parser.add_argument(
        "--log_dir",
        default="experiments/surrogate_models",
        type=str,
        help="Log directory",
    )
    parser.add_argument("--seed", type=int, default=6, help="Seed")
    parser.add_argument("--data_splits_root", default=None, type=str, help="path to dir containing data splits")
    parser.add_argument(
        "--ensemble", action="store_true", default=False, help="Ensemble"
    )
    args = parser.parse_args()
    # Create log directory
    if args.device is not None:
        assert args.metric is not None, f"Please specify metric for {args.device}"
        args.device = args.device.lower()
        args.metric = args.metric.lower()
        args.log_dir = os.path.join(args.log_dir, args.model, args.device, args.metric)
    else:
        args.log_dir = os.path.join(args.log_dir, args.model)
    # num_samples = 5000
    # fold = f"model_{num_samples}samples"
    log_dir = os.path.join(
        args.log_dir, "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.seed)
        # args.log_dir, "{}-{}".format(fold, args.seed)
    )
    os.makedirs(log_dir)

    data_config = json.load(open(args.data_config_path, 'r'))
    model_configspace = config_space_json_r_w.read(open(args.model_config_path, 'r').read())
    #model_configspace = utils.get_model_configspace(args.model)
    model_config = model_configspace.get_default_configuration().get_dictionary()
    model_config['model'] = args.model
    model_config['device'] = args.device
    model_config['metric'] = args.metric

    '''
    m_config = dict()
    for hyp in model_config["hyperparameters"]:
        m_config[hyp["name"]] = (
            hyp["default"] if "default" in hyp.keys() else hyp["value"]
        )
    '''
    #print(m_config)

    # Instantiate surrogate model
    if args.ensemble:
        surrogate_model = Ensemble(
            member_model_name=args.model,
            data_root=args.dataset_root,
            log_dir=log_dir,
            starting_seed=args.seed,
            model_config=model_config,
            data_config=data_config,
            ensemble_size=5,
            device=args.device,
            metric=args.metric
        )
    else:
        surrogate_model = utils.model_dict[args.model](
            data_root=args.dataset_root,
            log_dir=log_dir,
            seed=args.seed,
            model_config=model_config,
            data_config=data_config,
            device=args.device,
            metric=args.metric
        )

    if args.data_splits_root is not None:
        train_paths = json.load(open(os.path.join(args.data_splits_root, "train_paths.json"), "r"))
        val_paths = json.load(open(os.path.join(args.data_splits_root, "val_paths.json"), "r"))
        test_paths = json.load(open(os.path.join(args.data_splits_root, "test_paths.json"), "r"))

        print('=()=()='*10)
        print(len(train_paths), len(val_paths))

        cross_val_paths = train_paths + val_paths
        k_fold = KFold(n_splits=9, shuffle=True, random_state=args.seed)
        splits = list(k_fold.split(cross_val_paths))

        train_inds, val_inds = splits[args.seed % len(splits)]

        print('=)(=)(='*10)
        print(len(train_inds), len(val_inds))

        surrogate_model.train_paths = list(np.array(cross_val_paths)[train_inds])
        surrogate_model.val_paths = list(np.array(cross_val_paths)[val_inds])
        surrogate_model.test_paths = test_paths

    # Train and validate the model on the available data
    surrogate_model.train()
    if len(surrogate_model.test_paths) > 0:
        surrogate_model.test()
    # Save the model
    surrogate_model.save()


if __name__ == "__main__":
    train_surrogate_model()
