import os
import time
from ConfigSpace.read_and_write import json as cs_json
import torch
import torch.optim as optim
import numpy as np
import logging
from multiprocessing import Process, Queue, Lock
import multiprocessing
import csv
import pickle

from utils_anb.controller import Controller
from utils_anb.utils_nasnet import actions_indices_to_config
from utils_anb.utils_nasnet import pareto_frontier

multiprocessing.set_start_method("spawn", force=True)


def consume_tacc(worker, lock, accs_queue, action_indices_queue, design_queue):
    # worker.LaunchTrainJob(actions_index)
    worker.PollWBArtifact(lock)
    lock.acquire()
    worker_acc = worker.acc
    worker_actions_index = worker.actions_index.tolist()
    worker_design = worker.design
    accs_queue.put(worker_acc)
    action_indices_queue.put(worker_actions_index)
    design_queue.put(worker_design)
    lock.release()
    torch.cuda.empty_cache()
    del worker


class PolicyGradient(object):
    def __init__(self, args, ss_configspace_path, surrogate_model, device):
        self.args = args
        self.device = device
        with open(ss_configspace_path, "r") as fh:
            json_string = fh.read()
        self.ss_configspace = cs_json.read(json_string)
        self.surrogate_model = surrogate_model

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.controller = Controller(
            args.embedding_size, device=device, seed=self.args.seed
        ).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight
        self.start_ep = 0
        self.workers_top20 = []
        if args.resume_path is not None:
            _res = torch.load(args.resume_path)
            self.start_ep = _res["epoch"] + 1
            self.controller.load_state_dict(_res["controller_state"])
            self.adam.load_state_dict(_res["optimizer_state"])
            self.baseline = _res["baseline"]
            self.workers_top20 = _res["workers_top20"]
            print(f"[RESUME] Search resumed from Epoch {self.start_ep}")

    def multi_solve_environment(self, csv_path, exp_dir):
        from utils_anb.worker import (
            WorkerTACC,
        )  # This corresponds to a worker that submits a train job on our in-house cluster. This would need to be adapted to the users own slurm/cloud servers.

        submit_lock = Lock()
        csv_hres = os.path.join(exp_dir, "results_granular.csv")
        with open(csv_path, "a+") as fh:
            writer = csv.writer(fh)
            for arch_epoch in range(self.start_ep, self.arch_epochs):
                ep_st = time.time()
                accs_queue = Queue()
                actions_indices_queue = Queue()
                design_queue = Queue()
                processes = []
                episode = 0
                while episode < self.episodes:
                    # actions_p, actions_log_p, actions_index = self.controller.sample()
                    # actions_p = actions_p.cpu().numpy().tolist()
                    # actions_log_p = actions_log_p.cpu().numpy().tolist()
                    # actions_index = actions_index.cpu().numpy().tolist()
                    worker = WorkerTACC(
                        "PG", arch_epoch, episode, self.ss_configspace, self.controller
                    )
                    worker.LaunchTrainJob(submit_lock)
                    process = Process(
                        target=consume_tacc,
                        args=(
                            worker,
                            submit_lock,
                            accs_queue,
                            actions_indices_queue,
                            design_queue,
                        ),
                    )
                    process.start()
                    processes.append(process)
                    episode += 1
                for process in processes:
                    process.join()
                ep_time = time.time() - ep_st
                worker_accs = []
                worker_actions_indices = []
                worker_designs = []
                for episode in range(self.episodes):
                    acc = accs_queue.get(block=False)
                    ac_index = actions_indices_queue.get(block=False)
                    des = design_queue.get(block=False)
                    # worker.actions_p = torch.Tensor(worker.actions_p).to(self.device)
                    ac_index = torch.LongTensor(ac_index).to(self.device)
                    worker_accs.append(acc)
                    worker_actions_indices.append(ac_index)
                    worker_designs.append(des)

                with open(csv_hres, "a+") as fh1:
                    writer_hres = csv.writer(fh1)
                    for episode, (acc, design) in enumerate(
                        zip(worker_accs, worker_designs)
                    ):
                        if self.baseline is None:
                            self.baseline = acc
                        else:
                            self.baseline = (
                                self.baseline * self.baseline_weight
                                + acc * (1 - self.baseline_weight)
                            )
                        des = design
                        design_flat = [item for sublist in des for item in sublist]
                        row = [arch_epoch, episode, ep_time, acc]
                        row.extend(design_flat)
                        writer_hres.writerow(row)
                        fh1.flush()

                # sort worker retain top20
                current_accs = worker_accs
                workers_total = self.workers_top20 + current_accs
                # workers_total.sort(key=lambda worker: worker.acc, reverse=True)
                workers_total.sort(reverse=True)
                self.workers_top20 = workers_total[:20]
                top1_acc = self.workers_top20[0]
                top5_avg_acc = np.mean([_acc for _acc in self.workers_top20[:5]])
                top20_avg_acc = np.mean([_acc for _acc in self.workers_top20])

                loss = 0
                for acc, actions_index in zip(worker_accs, worker_actions_indices):
                    actions_p, actions_log_p = self.controller.get_p(actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, acc, self.baseline)

                loss /= len(worker_accs)
                logging.info(
                    "arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f} baseline {:.4f} Loss {:.4f}".format(
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top20_avg_acc,
                        self.baseline,
                        loss,
                    )
                )

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()
                writer.writerow(
                    [
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top20_avg_acc,
                        self.baseline,
                        loss,
                    ]
                )
                fh.flush()
                self.save_controller_params(arch_epoch, exp_dir)

    def cal_loss(self, actions_p, actions_log_p, worker_acc, baseline):
        reward = worker_acc - baseline
        policy_loss = -1 * torch.sum(actions_log_p * reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight
        return policy_loss + entropy_bonus

    def save_controller_params(self, epoch, exp_dir):
        save_file = os.path.join(exp_dir, "controller.pth")
        torch.save(
            {
                "epoch": epoch,
                "controller_state": self.controller.state_dict(),
                "optimizer_state": self.adam.state_dict(),
                "baseline": self.baseline,
                "workers_top20": self.workers_top20,
            },
            save_file,
            _use_new_zipfile_serialization=False,
        )


class PolicyGradientSimulated(object):
    # Simulated uni-objective REINFORCE using the accuracy surrogate
    def __init__(self, args, ss_configspace_path, surrogate_model, device):
        self.args = args
        self.device = device
        with open(ss_configspace_path, "r") as fh:
            json_string = fh.read()
        self.ss_configspace = cs_json.read(json_string)
        self.surrogate_model = surrogate_model

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.controller = Controller(
            args.embedding_size, device=device, seed=self.args.seed
        ).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight
        self.start_ep = 0
        self.workers_top20 = []

    def multi_solve_environment(self, csv_path, exp_dir):
        csv_hres = os.path.join(exp_dir, "results_granular.csv")
        top_design = None
        current_best_acc = 0
        with open(csv_path, "a+") as fh:
            writer = csv.writer(fh)
            for arch_epoch in range(self.start_ep, self.arch_epochs):
                ep_st = time.time()
                episode = 0
                actions_indices = []
                actions_configs = []
                designs = []
                while episode < self.episodes:
                    actions_p, actions_log_p, actions_index = self.controller.sample()
                    actions_config, design = actions_indices_to_config(
                        actions_index, self.ss_configspace
                    )
                    actions_indices.append(actions_index)
                    actions_configs.append(actions_config)
                    designs.append(design)
                    episode += 1
                worker_accs, _ = self.surrogate_model.query(actions_configs)
                worker_designs = designs
                worker_actions_indices = actions_indices
                ep_time = time.time() - ep_st
                with open(csv_hres, "a+") as fh1:
                    writer_hres = csv.writer(fh1)
                    for episode, (acc, design) in enumerate(
                        zip(worker_accs, worker_designs)
                    ):
                        if self.baseline is None:
                            self.baseline = acc
                        else:
                            self.baseline = (
                                self.baseline * self.baseline_weight
                                + acc * (1 - self.baseline_weight)
                            )
                        des = design
                        design_flat = [item for sublist in des for item in sublist]
                        row = [arch_epoch, episode, ep_time, acc]
                        row.extend(design_flat)
                        writer_hres.writerow(row)
                        fh1.flush()

                # sort worker retain top20
                for _ep, (_acc, _design) in enumerate(zip(worker_accs, worker_designs)):
                    if current_best_acc < _acc:
                        current_best_acc = _acc
                        top_design = _design
                avg_worker_accs = sum(worker_accs) / len(worker_accs)
                best_worker_acc = max(worker_accs)
                worker_accs = worker_accs.tolist()
                current_accs = worker_accs
                workers_total = self.workers_top20 + current_accs
                # workers_total.sort(key=lambda worker: worker.acc, reverse=True)
                workers_total.sort(reverse=True)
                self.workers_top20 = workers_total[:20]
                top1_acc = self.workers_top20[0]
                top5_avg_acc = np.mean([_acc for _acc in self.workers_top20[:5]])
                top20_avg_acc = np.mean([_acc for _acc in self.workers_top20])

                loss = 0
                for acc, actions_index in zip(worker_accs, worker_actions_indices):
                    actions_p, actions_log_p = self.controller.get_p(actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, acc, self.baseline)

                loss /= len(worker_accs)
                logging.info(
                    "arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f} baseline {:.4f} Loss {:.4f}".format(
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top20_avg_acc,
                        self.baseline,
                        loss,
                    )
                )

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()
                writer.writerow(
                    [
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top20_avg_acc,
                        self.baseline,
                        loss,
                    ]
                )
                fh.flush()
                self.save_controller_params(arch_epoch, exp_dir)

    def cal_loss(self, actions_p, actions_log_p, worker_acc, baseline):
        reward = worker_acc - baseline
        policy_loss = -1 * torch.sum(actions_log_p * reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus

    def save_controller_params(self, epoch, exp_dir):
        save_file = os.path.join(exp_dir, "controller.pth")
        torch.save(
            {
                "epoch": epoch,
                "controller_state": self.controller.state_dict(),
                "optimizer_state": self.adam.state_dict(),
                "baseline": self.baseline,
                "workers_top20": self.workers_top20,
            },
            save_file,
            _use_new_zipfile_serialization=False,
        )


class PolicyGradientSimulatedMO(object):
    # Bi-objective REINFORCE for accuracy-throughput/latency optimization
    # Takes accuracy and second objetive surrogate instances as inputs.
    def __init__(
        self, args, ss_configspace_path, acc_surrogate, biobj_surrogate, device
    ):
        self.args = args
        self.device = device
        with open(ss_configspace_path, "r") as fh:
            json_string = fh.read()
        self.ss_configspace = cs_json.read(json_string)
        self.acc_surrogate = acc_surrogate
        self.biobj_surrogate = biobj_surrogate
        self.target_biobj = args.target_biobj

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.controller = Controller(
            args.embedding_size, device=device, seed=self.args.seed
        ).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight
        self.start_ep = 0
        self.workers_top20_acc = []
        self.workers_top20_biobj = []

        self.plot_freq = 20

    def obtain_perf(self):
        accs, biobj = [], []
        actions_configs, actions_indices, designs = [], [], []

        for _episode in range(self.episodes):
            actions_p, actions_log_p, actions_index = self.controller.sample()
            actions_config, design = actions_indices_to_config(
                actions_index, self.ss_configspace
            )
            actions_indices.append(actions_index)
            actions_configs.append(actions_config)
            designs.append(design)

        accs, _ = self.acc_surrogate.query(actions_configs)
        biobj, _ = self.biobj_surrogate.query(actions_configs)
        # accs = self.acc_surrogate.query_noise(actions_configs)
        # biobj = self.biobj_surrogate.query_noise(actions_configs)
        accs = accs.tolist()
        biobj = biobj.tolist()
        return accs, biobj, actions_indices, designs

    def multi_solve_environment(self, csv_path, exp_dir):
        csv_hres = os.path.join(exp_dir, "results_granular.csv")
        all_accs = []
        all_biobjs = []
        all_designs = []
        with open(csv_path, "a+") as fh:
            writer = csv.writer(fh)
            for arch_epoch in range(self.start_ep, self.arch_epochs):
                (
                    worker_accs,
                    worker_biobjs,
                    worker_actions_indices,
                    worker_designs,
                ) = self.obtain_perf()

                with open(csv_hres, "a+") as fh1:
                    writer_hres = csv.writer(fh1)
                    for episode, (acc, metric_biobj, design) in enumerate(
                        zip(worker_accs, worker_biobjs, worker_designs)
                    ):
                        if self.baseline is None:
                            self.baseline = self.calc_reward(
                                acc, metric_biobj, self.target_biobj
                            )
                        else:
                            self.baseline = (
                                self.baseline * self.baseline_weight
                                + self.calc_reward(acc, metric_biobj, self.target_biobj)
                                * (1 - self.baseline_weight)
                            )
                        des = design
                        design_flat = [item for sublist in des for item in sublist]
                        row = [arch_epoch, episode, acc]
                        row.extend(design_flat)
                        writer_hres.writerow(row)
                        fh1.flush()

                all_accs.extend(worker_accs)
                all_biobjs.extend(worker_biobjs)
                all_designs.extend(worker_designs)
                if (
                    arch_epoch % self.plot_freq == 0
                    or arch_epoch == self.arch_epochs - 1
                ):
                    accs_pareto, size_pareto, lst = pareto_frontier(
                        np.array([all_accs, all_biobjs])
                    )
                    fname = f"scatter_{arch_epoch}"
                    fname = (
                        fname + ".png"
                        if arch_epoch != self.arch_epochs - 1
                        else fname + ".pdf"
                    )
                    # baseline_eff = (254, 65.74) if arch_epoch == self.arch_epochs - 1 else None
                    # label_points = None if arch_epoch != self.arch_epochs - 1 else [(285.5571289, 66.83581543), (399.8745117, 66.20720673), (540.2183838, 65.12088776)]
                    label_points = (
                        None
                        if arch_epoch != self.arch_epochs - 1
                        else [(3302.111328, 67.07880402), (3735.628906, 66.35800934)]
                    )
                    # label_points = None if arch_epoch != self.arch_epochs - 1 else [(1465.461914, 66.91249084), (1316.0687, 67.29836)]
                    # label_points = None if arch_epoch != self.arch_epochs - 1 else [(8011.878418, 67.06536865), (12057.91797, 66.24349213)]
                    # label_points = None if arch_epoch != self.arch_epochs - 1 else [(6000.564453, 66.97509766), (8079.51416, 66.24349213)]
                    with open(os.path.join(exp_dir, "search_res.pkl"), "wb") as pp:
                        pickle.dump(
                            [
                                all_biobjs,
                                all_accs,
                                size_pareto,
                                accs_pareto,
                                label_points,
                            ],
                            pp,
                        )
                    """
                    plot_scatter_pareto(
                        all_biobjs,
                        all_accs,
                        np.array([size_pareto, accs_pareto]),
                        os.path.join(exp_dir, fname),
                        label_top5=False, # arch_epoch == self.arch_epochs - 1,
                        label_points=label_points,
                        baseline_effnet=None, #baseline_eff,
                    )
                    """
                    top_designs = {}
                    for i in lst:
                        top_designs[all_accs[i]] = [all_biobjs[i], all_designs[i]]
                    pickle.dump(
                        top_designs,
                        open(
                            os.path.join(exp_dir, f"pareto_designs{arch_epoch}.pkl"),
                            "wb",
                        ),
                    )
                current_accs = worker_accs
                current_biobjs = worker_biobjs
                best_acc = max(current_accs)
                mean_acc = sum(current_accs) / len(current_accs)
                best_biobj = max(current_biobjs)
                mean_biobj = sum(current_biobjs) / len(current_biobjs)
                workers_total_acc = self.workers_top20_acc + current_accs
                workers_total_biobj = self.workers_top20_biobj + current_biobjs

                sort_indices = np.argsort(-np.array(workers_total_acc))
                workers_total_acc = list(np.array(workers_total_acc)[sort_indices])
                workers_total_biobj = list(np.array(workers_total_biobj)[sort_indices])

                self.workers_top20_acc = workers_total_acc[:20]
                self.workers_top20_biobj = workers_total_biobj[:20]
                top1_acc = self.workers_top20_acc[0]
                top1_biobj = self.workers_top20_biobj[0]
                top5_avg_acc = np.mean([_acc for _acc in self.workers_top20_acc[:5]])
                top5_avg_biobj = np.mean(
                    [_biobj for _biobj in self.workers_top20_biobj[:5]]
                )

                loss = 0
                for acc, metric_biobj, actions_index in zip(
                    worker_accs, worker_biobjs, worker_actions_indices
                ):
                    actions_p, actions_log_p = self.controller.get_p(
                        actions_index.to(self.device)
                    )
                    cur_reward = self.calc_reward(acc, metric_biobj, self.target_biobj)
                    loss += self.cal_loss(
                        actions_p, actions_log_p, cur_reward, self.baseline
                    )

                loss /= len(worker_accs)
                logging.info(
                    "arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top1_biobj {:.4f} top5_biobj {:.4f} baseline {:.4f} Loss {:.4f}".format(
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top1_biobj,
                        top5_avg_biobj,
                        self.baseline,
                        loss,
                    )
                )

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()
                writer.writerow(
                    [
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top1_biobj,
                        top5_avg_biobj,
                        self.baseline,
                        loss,
                    ]
                )
                fh.flush()
                self.save_controller_params(arch_epoch, exp_dir)

    def calc_reward(self, worker_acc, biobj_metric, target_biobj):
        reward = worker_acc * pow((biobj_metric / target_biobj), 0.07)
        # reward = worker_acc * pow((target_biobj / biobj_metric), 0.07)
        return reward

    def cal_loss(self, actions_p, actions_log_p, worker_acc, baseline):
        reward = worker_acc - baseline
        policy_loss = -1 * torch.sum(actions_log_p * reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight
        return policy_loss + entropy_bonus

    def save_controller_params(self, epoch, exp_dir):
        save_file = os.path.join(exp_dir, "controller.pth")
        torch.save(
            {
                "epoch": epoch,
                "controller_state": self.controller.state_dict(),
                "optimizer_state": self.adam.state_dict(),
                "baseline": self.baseline,
                "workers_top20_acc": self.workers_top20_acc,
                "workers_top20_biobj": self.workers_top20_biobj,
            },
            save_file,
            _use_new_zipfile_serialization=False,
        )
