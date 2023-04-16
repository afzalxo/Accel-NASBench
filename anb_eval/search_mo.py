import numpy as np
import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys

import wandb

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import accelnb as anb

parser = argparse.ArgumentParser("search-imagenet")
parser.add_argument("--resume_path", type=str, default=None)
# architecture
parser.add_argument("--arch_epochs", type=int, default=20)
parser.add_argument("--arch_lr", type=float, default=3.5e-4)
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--entropy_weight", type=float, default=1e-5)
parser.add_argument("--baseline_weight", type=float, default=0.95)
parser.add_argument("--embedding_size", type=int, default=32)
parser.add_argument("--algorithm", type=str, choices=["PPO", "PG", "RS"], default="RS")
parser.add_argument("--simulated", action="store_true", default=True)
# PPO
parser.add_argument("--ppo_epochs", type=int, default=10)
parser.add_argument("--clip_epsilon", type=float, default=0.2)

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2, help="random seed")

# TODO: Remove dependency on WandB
parser.add_argument("--wandb_project", type=str, help="Name of wandb project")
parser.add_argument("--wandb_user", type=str, help="Name of wandb entity")

# Bi Objective
parser.add_argument("--target_biobj", type=float, default=400.0, help="Target throughput")
parser.add_argument("--device", type=str, default="vck190", help="Target device")
parser.add_argument("--metric", type=str, default="throughput", help="Second objective")
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(str(args.gpu)))
        cudnn.benchmark = True
        cudnn.enable = True
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")

    search_device = args.device
    search_metric = args.metric
    search_type = "SIM" if args.simulated else "REAL"
    exp_dir = "logs/simulated/search_{}_{}_{}_{}_seed{}_{}".format(
        args.algorithm, search_type, search_device, search_metric, args.seed, time.strftime("%Y%m%d-%H%M%S")
    )
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    csv_path = os.path.join(exp_dir, "result.csv")
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )
    fh = logging.FileHandler(os.path.join(exp_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", args)

    wandb_con = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_user,
        name=f"search-{args.algorithm}",
        group=f"search-{args.algorithm}-{search_type}",
    )
    args.wandb_con = wandb_con

    configspace_path = "configspace/configspace.json"
    acc_surrogate = anb.ANBEnsemble("xgb", seed=None).load_ensemble()
    biobj_surrogate = anb.ANBEnsemble(
        "xgb", device=search_device, metric=search_metric, seed=None
    ).load_ensemble()
    if args.algorithm == "PG":
        from nas_optimizers.policy_gradient import PolicyGradientSimulatedMO
        pg = PolicyGradientSimulatedMO(
            args, configspace_path, acc_surrogate, biobj_surrogate, device
        )
        pg.multi_solve_environment(csv_path, exp_dir, wandb_con)
    elif args.algorithm == "RS":
        from nas_optimizers.random_search import RandomSearchSimulatedMO
        rs = RandomSearchSimulatedMO(
            args, configspace_path, acc_surrogate, biobj_surrogate
        )
        rs.multi_solve_environment(csv_path, exp_dir, wandb_con)


if __name__ == "__main__":
    main()
