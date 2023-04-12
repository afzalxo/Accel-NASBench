from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write import json as config_space_json_r_w
from smac import HyperparameterOptimizationFacade, Scenario

import json
import os
import time
import argparse

import utils

train_paths = "configs/data_splits/default_split/train_paths.json"
val_paths = "configs/data_splits/default_split/val_paths.json"
test_paths = "configs/data_splits/default_split/test_paths.json"


class HPOModel:
    def __init__(
        self,
        model: str,
        dataset_root: str,
        log_dir: str,
        data_config_path: str,
        model_config_path: str,
        seed: int = 0,
        device=None,
        metric=None
    ):
        self.dataset_root = dataset_root
        self.data_config = json.load(open(data_config_path, "r"))
        self.model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())# utils.get_model_configspace(model)
        self.device = device
        self.metric = metric
        self.log_dir = log_dir

    def get_model_configspace(self) -> ConfigurationSpace:
        return self.model_configspace

    def do_hpo(self, model_config: Configuration, seed: int = 0) -> float:
        model_config = model_config.get_dictionary()
        surrogate_model = utils.model_dict[args.model](
            data_root=self.dataset_root,
            log_dir=self.log_dir,
            seed=seed,
            model_config=model_config,
            data_config=self.data_config,
            device=self.device,
            metric=self.metric
        )
        # surrogate_model.train_paths = json.load(open(train_paths, 'r'))
        # surrogate_model.val_paths = json.load(open(val_paths, 'r'))
        # surrogate_model.test_paths = json.load(open(test_paths, 'r'))
        valid_metrics = surrogate_model.train()
        # print("=*=*=*=" * 10)
        # print(valid_metrics["rmse"])
        return valid_metrics["rmse"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Surrogate HPO")
    parser.add_argument("--dataset_root", type=str, help="Path to dataset root dir")
    parser.add_argument("--model", type=str, help="Surrogate model")
    parser.add_argument("--data_config_path", type=str, help="Data config path")
    parser.add_argument("--model_config_path", type=str, help="Model config path")
    parser.add_argument("--device", default=None, help="Device, None if acc")
    parser.add_argument("--metric", default=None, help="Metric, None if acc")
    parser.add_argument(
        "--log_dir",
        default="experiments/hpo_surrogate",
        type=str,
        help="Log directory",
    )
    parser.add_argument("--seed", type=int, default=6, help="Seed")
    args = parser.parse_args()

    if args.device is not None:
        assert args.metric is not None, f"Missing metric for {args.device}"
        args.log_dir = os.path.join(args.log_dir, args.model, args.device, args.metric)
    else:
        args.log_dir = os.path.join(args.log_dir, args.model)
    args.log_dir = os.path.join(
        args.log_dir, "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.seed)
    )
    os.makedirs(args.log_dir)

    hpo_model = HPOModel(
        args.model,
        args.dataset_root,
        args.log_dir,
        args.data_config_path,
        args.model_config_path,
        seed=args.seed,
        device=args.device,
        metric=args.metric,
    )
    scenario = Scenario(
        hpo_model.get_model_configspace(), deterministic=True, walltime_limit=600# n_trials=200
    )
    smac = HyperparameterOptimizationFacade(scenario, hpo_model.do_hpo)
    incumbent = smac.optimize()
    print('Final Incumbent:')
    print(incumbent)
    save_file = os.path.join(args.log_dir, f'{args.model}_{args.device}_{args.metric}_config.json')
    json.dump(incumbent.get_dictionary(), open(save_file, 'w'))
