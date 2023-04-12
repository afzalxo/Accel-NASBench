import os
import json
from pathlib import Path
import utils

from typing import Optional

from ensemble import Ensemble
from model_downloader import download_models


class ANBEnsemble:
    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        metric: Optional[str] = None,
        seed: Optional[int] = None,
        subdir: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.metric = metric
        self.seed = seed
        self.subdir = subdir
        if "accel" not in model and device is not None:
            self.model = model + "_accel"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_0_9_paths = os.path.join(current_dir, "anb_models_0_9")
        if not os.path.exists(models_0_9_paths):
            download_models(version="0.9", delete_zip=True, download_dir=current_dir)
        if self.device is not None and self.subdir == None:
            models_0_9_paths = os.path.join(
                models_0_9_paths, self.model, self.device, self.metric
            )
        elif self.subdir is None:
            models_0_9_paths = os.path.join(models_0_9_paths, self.model)
        elif self.subdir is not None:
            models_0_9_paths = os.path.join(
                current_dir, subdir
            )
            print(f"Loading ablation model from subdir: {models_0_9_paths}")
        self.parent_dir = models_0_9_paths

    def load_model(self):
        member_dirs = [
            os.path.dirname(filename)
            for filename in Path(self.parent_dir).rglob("*surrogate_model.model")
        ]
        data_config = json.load(
            open(os.path.join(member_dirs[0], "data_config.json"), "r")
        )
        model_config = json.load(
            open(os.path.join(member_dirs[0], "model_config.json"), "r")
        )
        if self.seed is None:
            self.seed = data_config["seed"]
        model_name = model_config["model"]
        device = model_config["device"]
        metric = model_config["metric"]
        print(f"Loading {model_name} model, Device={device}, Metric={metric}")
        surrogate_model = utils.model_dict[model_name](
            data_root="None",
            log_dir=self.parent_dir,
            seed=self.seed,
            model_config=model_config,
            data_config=data_config,
            device=device,
            metric=metric,
        )
        surrogate_model.load(
            model_path=os.path.join(member_dirs[0], "surrogate_model.model")
        )
        return surrogate_model

    def load_ensemble(self) -> Ensemble:
        member_dirs = [
            os.path.dirname(filename)
            for filename in Path(self.parent_dir).rglob("*surrogate_model.model")
        ]
        data_config = json.load(
            open(os.path.join(member_dirs[0], "data_config.json"), "r")
        )
        model_config = json.load(
            open(os.path.join(member_dirs[0], "model_config.json"), "r")
        )
        if self.seed is None:
            self.seed = data_config["seed"]
        model_name = model_config["model"]
        device = model_config["device"]
        metric = model_config["metric"]
        print(
            f"Loading {model_name} ensemble with {len(member_dirs)} members, Device={device}, Metric={metric}"
        )
        surrogate_model = Ensemble(
            member_model_name=model_config["model"],
            data_root="None",
            log_dir=self.parent_dir,
            starting_seed=self.seed,
            model_config=model_config,
            data_config=data_config,
            ensemble_size=len(member_dirs),
            init_ensemble=False,
            device=device,
            metric=metric,
        )
        surrogate_model.load(
            model_paths=member_dirs,
        )
        return surrogate_model
