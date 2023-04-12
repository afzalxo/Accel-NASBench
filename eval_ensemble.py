import sys
import os
import json
import logging
import argparse
from pathlib import Path
import time

from ensemble import Ensemble

data_root = "dataset_parsers/datasets_final_jsons/json/"
train_paths = "configs/data_splits/default_split/train_paths.json"
val_paths = "configs/data_splits/default_split/val_paths.json"
test_paths = "configs/data_splits/default_split/test_paths.json"


def load_ensemble(parent_dir):
    member_dirs = [
        os.path.dirname(filename)
        for filename in Path(parent_dir).rglob("*surrogate_model.model")
    ]
    data_config = json.load(open(os.path.join(member_dirs[0], "data_config.json"), "r"))
    model_config = json.load(
        open(os.path.join(member_dirs[0], "model_config.json"), "r")
    )
    log_dir = os.path.join(
        parent_dir, "exp_ensemble-{}".format(time.strftime("%Y%m%d-%H%M%S"))
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    model_name = model_config["model"]
    device = model_config["device"]
    metric = model_config["metric"]
    print(
        f"Loading {model_name} ensemble with {len(member_dirs)} members, Device={device}, Metric={metric}"
    )
    surrogate_model = Ensemble(
        member_model_name=model_config["model"],
        data_root=data_root,
        log_dir=log_dir,
        starting_seed=data_config["seed"],
        model_config=model_config,
        data_config=data_config,
        ensemble_size=len(member_dirs),
        init_ensemble=False,
        device=device,
        metric=metric,
    )
    surrogate_model.load(
        model_paths=member_dirs,
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
    )
    return surrogate_model


parser = argparse.ArgumentParser("Surrogate Ensemble Evaluate")
parser.add_argument("--device", type=str, help="Device", default=None)
parser.add_argument("--metric", type=str, help="Metric", default=None)
parser.add_argument("--model", type=str, help="Surrogate model")
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
surr_models_dir = os.path.join(current_dir, "experiments", "surrogate_models")

device = args.device
metric = args.metric
_model_name = args.model

if device is None:
    model_name = _model_name
    model_dir = os.path.join(surr_models_dir, model_name)
else:
    if 'accel' not in _model_name:
        model_name = _model_name + "_accel"
    else:
        model_name = _model_name
    model_dir = os.path.join(surr_models_dir, model_name, device, metric)

acc_model = load_ensemble(model_dir)
val_metrics = acc_model.validate_ensemble(apply_noise=False)[0]
logging.info('val metrics: %s', val_metrics)
test_metrics = acc_model.test_ensemble(apply_noise=False)[0]
logging.info('test metrics: %s', test_metrics)
