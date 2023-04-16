import os
import time
from ConfigSpace.read_and_write import json as cs_json
import numpy as np
import torch
import torch.optim as optim
import logging
from multiprocessing import Process, Queue
import multiprocessing
import csv
import pickle
import wandb

from AlgoUtils.Worker import WorkerTACC
from AlgoUtils.controller import Controller
from AlgoUtils.utils_nasnet import actions_indices_to_config
from AlgoUtils.utils_nasnet import plot_scatter_pareto
from AlgoUtils.utils_nasnet import pareto_frontier

multiprocessing.set_start_method('spawn', force=True)


class PPO(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.arch_epochs = args.arch_epochs
        self.arch_lr = args.arch_lr
        self.episodes = args.episodes
        self.entropy_weight = args.entropy_weight

        self.ppo_epochs = args.ppo_epochs

        self.controller = Controller(args, device=device).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight

        self.clip_epsilon = 0.2

    def multi_solve_environment(self):
        workers_top20 = []

        for arch_epoch in range(self.arch_epochs):
            results_queue = Queue()
            processes = []

            for episode in range(self.episodes):
                actions_p, actions_log_p, actions_index = self.controller.sample()
                actions_p = actions_p.cpu().numpy().tolist()
                actions_log_p = actions_log_p.cpu().numpy().tolist()
                actions_index = actions_index.cpu().numpy().tolist()

                if episode < self.episodes // 3:
                    worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:0')
                elif self.episodes // 3 <= episode < 2 * self.episodes // 3:
                    worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:1')
                else:
                    worker = Worker(actions_p, actions_log_p, actions_index, self.args, 'cuda:3')

                process = Process(target=consume, args=(worker, results_queue))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

            workers = []
            for episode in range(self.episodes):
                worker = results_queue.get()
                worker.actions_p = torch.Tensor(worker.actions_p).to(self.device)
                worker.actions_index = torch.LongTensor(worker.actions_index).to(self.device)
                workers.append(worker)

            for episode, worker in enumerate(workers):
                if self.baseline == None:
                    self.baseline = worker.acc
                else:
                    self.baseline = self.baseline * self.baseline_weight + worker.acc * (1 - self.baseline_weight)

            # sort worker retain top20
            workers_total = workers_top20 + workers
            workers_total.sort(key=lambda worker: worker.acc, reverse=True)
            workers_top20 = workers_total[:20]
            top1_acc = workers_top20[0].acc
            top5_avg_acc = np.mean([worker.acc for worker in workers_top20[:5]])
            top20_avg_acc = np.mean([worker.acc for worker in workers_top20])
            logging.info('arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f} baseline {:.4f} '.format(
                arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc, self.baseline))
            for i in range(5):
                print(workers_top20[i].genotype)

            for ppo_epoch in range(self.ppo_epochs):
                loss = 0

                for worker in workers:
                    actions_p, actions_log_p = self.controller.get_p(worker.actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, worker, self.baseline)

                loss /= len(workers)
                logging.info('ppo_epoch {:0>3d} loss {:.4f} '.format(ppo_epoch, loss))

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()


    def solve_environment(self):
        for arch_epoch in range(self.arch_epochs):
            workers = []
            acc = 0

            for episode in range(self.episodes):
                actions_p, actions_log_p, actions_index = self.controller.sample()
                workers.append(Worker(actions_p, actions_log_p, actions_index, self.args, self.device))

            for episode, worker in enumerate(workers):
                worker.get_acc(self.train_queue, self.valid_queue)
                if self.baseline == None:
                    self.baseline = worker.acc
                else:
                    self.baseline = self.baseline * self.baseline_weight + worker.acc * (1 - self.baseline_weight)

                acc += worker.acc
                logging.info('episode {:0>3d} acc {:.4f} baseline {:.4f}'.format(episode, worker.acc, self.baseline))
            acc /= self.episodes
            logging.info('arch_epoch {:0>3d} acc {:.4f} '.format(arch_epoch, acc))

            for ppo_epoch in range(self.ppo_epochs):
                loss = 0

                for worker in workers:
                    actions_p, actions_log_p = self.controller.get_p(worker.actions_index)

                    loss += self.cal_loss(actions_p, actions_log_p, worker, self.baseline)

                loss /= len(workers)
                logging.info('ppo_epoch {:0>3d} loss {:.4f} '.format(ppo_epoch, loss))

                self.adam.zero_grad()
                loss.backward()
                self.adam.step()

    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance

    def cal_loss(self, actions_p, actions_log_p, worker, baseline):
        actions_importance = actions_p / worker.actions_p
        clipped_actions_importance = self.clip(actions_importance)
        reward = worker.acc - baseline
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus


class PPOSimulatedMO(object):
    def __init__(self, args, ss_configspace_path, acc_surrogate, biobj_surrogate, device):
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

        self.ppo_epochs = args.ppo_epochs

        self.controller = Controller(args.embedding_size, device=device).to(device)

        self.adam = optim.Adam(params=self.controller.parameters(), lr=self.arch_lr)

        self.baseline = None
        self.baseline_weight = self.args.baseline_weight
        self.workers_top20_acc = []
        self.workers_top20_biobj = []

        self.plot_freq = 5

        self.clip_epsilon = 0.2

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
        accs = accs.tolist()
        biobj = biobj.tolist()
        return accs, biobj, actions_indices, designs

    def multi_solve_environment(self, csv_path, exp_dir, wandb_con):
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
                    plot_scatter_pareto(
                        all_biobjs,
                        all_accs,
                        np.array([size_pareto, accs_pareto]),
                        os.path.join(exp_dir, f"scatter_{arch_epoch}.png"),
                        label_top5=arch_epoch == self.arch_epochs - 1,
                    )
                    top_designs = {}
                    for i in lst:
                        top_designs[all_accs[i]] = [all_biobjs[i], all_designs[i]]
                    pickle.dump(
                        top_designs,
                        open(os.path.join(exp_dir, "pareto_designs.pkl"), "wb"),
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
                for ppo_epoch in range(self.ppo_epochs):
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
                        "arch_epoch {:0>3d} ppo_epoch {:0>2d} top1_acc {:.4f} top5_avg_acc {:.4f} top1_biobj {:.4f} top5_biobj {:.4f} baseline {:.4f} Loss {:.4f}".format(
                            arch_epoch,
                            ppo_epoch,
                            top1_acc,
                            top5_avg_acc,
                            top1_biobj,
                            top5_avg_biobj,
                            self.baseline,
                            loss,
                        )
                    )

                    wandb_con.log(
                        {
                            "Loss": loss,
                            "PPO Epoch": ppo_epoch,
                            "Top-1 Acc": top1_acc,
                            "Top-5 Avg Acc": top5_avg_acc,
                            "Top-1 BiObj": top1_biobj,
                            "Top-5 Avg BiObj": top5_avg_biobj,
                            "Avg Epoch Acc": mean_acc,
                            "Avg Epoch BiObj": mean_biobj,
                        },
                        commit=True,
                    )
                    self.adam.zero_grad()
                    loss.backward()
                    self.adam.step()
                    writer.writerow(
                        [
                            arch_epoch,
                            ppo_epoch,
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

    def clip(self, actions_importance):
        lower = torch.ones_like(actions_importance).to(self.device) * (1 - self.clip_epsilon)
        upper = torch.ones_like(actions_importance).to(self.device) * (1 + self.clip_epsilon)

        actions_importance, _ = torch.min(torch.cat([actions_importance.unsqueeze(0), upper.unsqueeze(0)], dim=0), dim=0)
        actions_importance, _ = torch.max(torch.cat([actions_importance.unsqueeze(0), lower.unsqueeze(0)], dim=0), dim=0)

        return actions_importance

    def cal_loss(self, actions_p, actions_log_p, worker_acc, baseline):
        # FIXME: Need to go in depth
        actions_importance = actions_p / worker.actions_p
        clipped_actions_importance = self.clip(actions_importance)
        reward = worker.acc - baseline
        actions_reward = actions_importance * reward
        clipped_actions_reward = clipped_actions_importance * reward

        actions_reward, _ = torch.min(torch.cat([actions_reward.unsqueeze(0), clipped_actions_reward.unsqueeze(0)], dim=0), dim=0)
        policy_loss = -1 * torch.sum(actions_reward)
        entropy = -1 * torch.sum(actions_p * actions_log_p)
        entropy_bonus = -1 * entropy * self.entropy_weight

        return policy_loss + entropy_bonus
