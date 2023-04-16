import numpy as np
import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys
import random

anb_dir = os.path.dirname(os.getcwd())
sys.path.append(anb_dir)
import accelnb as anb

parser = argparse.ArgumentParser("search-imagenet")
parser.add_argument("--resume_path", type=str, default=None)
# architecture
parser.add_argument("--arch_epochs", type=int, default=100)
parser.add_argument("--arch_lr", type=float, default=3.5e-4)
parser.add_argument("--episodes", type=int, default=6)
parser.add_argument("--entropy_weight", type=float, default=1e-5)
parser.add_argument("--baseline_weight", type=float, default=0.95)
parser.add_argument("--embedding_size", type=int, default=32)
parser.add_argument("--algorithm", type=str, choices=["PG", "RS"], default="RS")
parser.add_argument("--simulated", action="store_true", default=True)

parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=2, help="random seed")
args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(str(args.gpu)))
        cudnn.benchmark = True
        cudnn.enable = True
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")

    search_type = "SIM" if args.simulated else "REAL"
    exp_dir = "./logs/simulated/search_{}_{}_seed{}_{}".format(
        args.algorithm, search_type, args.seed, time.strftime("%Y%m%d-%H%M%S")
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

    configspace_path = "configspace/configspace.json"
    surr_model = anb.ANBEnsemble("xgb", seed=args.seed).load_ensemble()
    print("Loaded ensemble...")

    if args.simulated:
        if args.algorithm == "RS":
            from SearchAlgorithms.random_search import RandomSearchSimulated

            rs = RandomSearchSimulated(
                configspace_path, args.arch_epochs, args.episodes, surr_model
            )
            rs.multi_solve_environment(csv_path)
        elif args.algorithm == "PG":
            from nas_optimizers.policy_gradient import (
                PolicyGradientSimulated,
            )

            pg = PolicyGradientSimulated(args, configspace_path, surr_model, device)
            pg.multi_solve_environment(csv_path, exp_dir)
    else:
        if args.algorithm == "PG":
            from nas_optimizers.policy_gradient import PolicyGradient
            if args.algorithm == "PG":
                pg = PolicyGradient(args, configspace_path, surr_model, device)
                pg.multi_solve_environment(csv_path, exp_dir)


if __name__ == "__main__":
    main()
