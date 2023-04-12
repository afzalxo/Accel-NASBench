import glob
import json
import os
import shutil
import time

import click
import matplotlib

import utils

matplotlib.use('Agg')


@click.command()
@click.option('--data_root', type=click.STRING, help='path to accel-nasbench root directory')
@click.option('--data_config_path', type=click.STRING, help='Path to config.json',
              default='configs/data_configs/nb_fpga.json')
@click.option('--splits_log_dir', type=click.STRING, help='Experiment directory',
              default='configs/data_splits/default_split')
@click.option('--seed', type=click.INT, help='seed for numpy, python, pytorch', default=6)
@click.option('--device', type=click.STRING, help='device', default='None')
@click.option('--metric', type=click.STRING, help='metric', default='None')
def create_data_splits(data_root, data_config_path, splits_log_dir, seed, device, metric):
    # Load config
    model = "xgb"
    model_config_path = None
    data_config = json.load(open(data_config_path, 'r'))

    # Create log directory
    log_dir = os.path.join("tmp", '{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), seed))
    os.makedirs(log_dir)
    os.makedirs(splits_log_dir, exist_ok=True)

    # Select model config to use
    if model_config_path is None:
        # Get model configspace
        model_configspace = utils.get_model_configspace(model)

        # Use default model config
        model_config = model_configspace.get_default_configuration().get_dictionary()
    else:
        model_config = json.load(open(model_config_path, 'r'))

    device = None if device == 'None' else device
    metric = None if metric == 'None' else device
    model_config['model'] = model
    model_config['device'] = device
    model_config['metric'] = metric

    # Instantiate surrogate model
    print("==> Instantiating surrogate")
    surrogate_model = utils.model_dict[model](data_root=data_root,
                                              log_dir=log_dir,
                                              seed=seed,
                                              model_config=model_config,
                                              data_config=data_config,
                                              device=device,
                                              metric=metric)

    # Save data splits
    print("==> Saving data splits")
    json.dump(surrogate_model.train_paths, open(os.path.join(splits_log_dir, "train_paths.json"), "w"))
    json.dump(surrogate_model.val_paths, open(os.path.join(splits_log_dir, "val_paths.json"), "w"))
    json.dump(surrogate_model.test_paths, open(os.path.join(splits_log_dir, "test_paths.json"), "w"))

    shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    create_data_splits()
