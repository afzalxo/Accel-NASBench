import os
import csv
import sys
from ConfigSpace.read_and_write import json as cs_json
import json
import numpy as np
import logging
from multiprocessing import Process, Queue
import multiprocessing
from utils_anb.utils_nasnet import configuration_to_searchable
from utils_anb.utils_nasnet import plot_scatter_pareto
from utils_anb.utils_nasnet import pareto_frontier
import pickle

multiprocessing.set_start_method("spawn", force=True)


def consume(worker, surrogate_model, actions_config, results_queue):
    worker.ObtainAcc(surrogate_model, actions_config)
    results_queue.put(worker)


class RandomSearch(object):
    def __init__(self, ss_configspace_path, arch_epochs, episodes, surrogate_model):
        with open(ss_configspace_path, "r") as fh:
            json_string = fh.read()
        self.surrogate_model = surrogate_model
        self.ss_configspace = cs_json.read(json_string)
        self.arch_epochs = arch_epochs
        self.episodes = episodes

    def random_sample(self):
        return self.ss_configspace.sample_configuration()

    def multi_solve_environment(self, csv_path):
        workers_top20 = []
        with open(csv_path, "a+") as fh:
            writer = csv.writer(fh)
            for arch_epoch in range(self.arch_epochs):
                results_queue = Queue()
                processes = []

                for episode in range(self.episodes):
                    actions_config = self.random_sample()
                    # worker = WorkerSimulatedRS()
                    # process = Process(target=consume, args=(worker, self.surrogate_model, actions_config, results_queue))
                    worker = WorkerTACC()
                    process = Process(
                        target=consume,
                        args=(
                            worker,
                            self.surrogate_model,
                            actions_config,
                            results_queue,
                        ),
                    )
                    process.start()
                    processes.append(process)

                for process in processes:
                    process.join()

                workers = []
                for episode in range(self.episodes):
                    worker = results_queue.get()
                    workers.append(worker)

                # sort worker retain top20
                workers_total = workers_top20 + workers
                workers_total.sort(key=lambda worker: worker.acc, reverse=True)
                workers_top20 = workers_total[:20]
                top1_acc = workers_top20[0].acc
                top5_avg_acc = np.mean([worker.acc for worker in workers_top20[:5]])
                top20_avg_acc = np.mean([worker.acc for worker in workers_top20])
                logging.info(
                    "arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top20_avg_acc {:.4f}".format(
                        arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc
                    )
                )
                writer.writerow([arch_epoch, top1_acc, top5_avg_acc, top20_avg_acc])
                fh.flush()
                # for i in range(5):
                #    print(workers_top20[i].genotype)


class RandomSearchSimulated(object):
    def __init__(self, ss_configspace_path, arch_epochs, episodes, surrogate_model):
        with open(ss_configspace_path, "r") as fh:
            json_string = fh.read()
        self.surrogate_model = surrogate_model
        self.ss_configspace = cs_json.read(json_string)
        self.arch_epochs = arch_epochs
        self.episodes = episodes
        self.algorithm = "RS-Sim"

    def random_sample(self):
        return self.ss_configspace.sample_configuration()

    def multi_solve_environment(self, csv_path):
        current_best_acc = 0
        top_design = None
        top5_accs = []
        with open(csv_path, "a+") as fh:
            writer = csv.writer(fh)
            for arch_epoch in range(self.arch_epochs):
                actions_configs = []
                designs = []
                for episode in range(self.episodes):
                    random_config = self.ss_configspace.sample_configuration()
                    actions_configs.append(random_config)
                    designs.append(configuration_to_searchable(random_config))

                accs, _ = self.surrogate_model.query(actions_configs)
                accs = accs.tolist()
                top1_acc_epoch = max(accs)
                acc_epoch_avg = sum(accs) / len(accs)

                for _ep, (_acc, _design) in enumerate(zip(accs, designs)):
                    if current_best_acc < _acc:
                        current_best_acc = _acc
                        top_design = _design
                # sort worker retain top20
                top_accs = top5_accs + accs
                top5_accs = sorted(top_accs, reverse=True)[:5]
                top1_acc = top_accs[0]
                top5_avg_acc = sum(top5_accs) / len(top5_accs)
                logging.info(
                    "arch_epoch {:0>3d} top1_acc {:.4f} top5_avg_acc {:.4f} top1_acc_epoch {:.4f} epoch_avg_acc {:.4f}".format(
                        arch_epoch,
                        top1_acc,
                        top5_avg_acc,
                        top1_acc_epoch,
                        acc_epoch_avg,
                    )
                )
                writer.writerow(
                    [arch_epoch, top1_acc, top5_avg_acc, top1_acc_epoch, acc_epoch_avg]
                )
                fh.flush()


class RandomSearchSimulatedMO(object):
    def __init__(
        self,
        args,
        ss_configspace_path,
        acc_surrogate,
        biobj_surrogate,
    ):
        with open(ss_configspace_path, "r") as fh:
            json_string = fh.read()
        self.ss_configspace = cs_json.read(json_string)
        self.acc_surrogate = acc_surrogate
        self.biobj_surrogate = biobj_surrogate

        self.arch_epochs = args.arch_epochs
        self.episodes = args.episodes
        self.algorithm = "RS-Sim"
        self.plot_freq = 5

    def random_sample(self):
        return self.ss_configspace.sample_configuration()

    def multi_solve_environment(self, csv_path, exp_dir):
        all_accs = []
        all_biobjs = []
        all_designs = []
        with open(csv_path, "a+") as fh:
            writer = csv.writer(fh)
            for arch_epoch in range(self.arch_epochs):
                actions_configs = []
                designs = []
                for episode in range(self.episodes):
                    random_config = self.ss_configspace.sample_configuration()
                    actions_configs.append(random_config)
                    designs.append(configuration_to_searchable(random_config))

                accs, _ = self.acc_surrogate.query(actions_configs)
                biobj, _ = self.biobj_surrogate.query(actions_configs)
                worker_accs = accs.tolist()
                worker_biobjs = biobj.tolist()

                all_accs.extend(worker_accs)
                all_biobjs.extend(worker_biobjs)
                all_designs.extend(designs)

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
                        open(
                            os.path.join(exp_dir, f"pareto_designs{arch_epoch}.pkl"),
                            "wb",
                        ),
                    )


def main():
    sys.path.append("/path/to/ANB/root/dir")
    import accelnb as anb
    configspace_path = (
        "<path/to/ANB/root/dir>/anb_eval/configspace/configspace.json"
    )
    surr_model = anb.ANBEnsemble("xgb").load_ensemble()
    sd = RandomSearchSimulated(configspace_path, 10, 10, surr_model)
    rs = sd.random_sample()
    print(rs.get_dictionary())
    with open("sample_json.json", "w") as fh:
        json.dump(rs.get_dictionary(), fh)
    exit(0)
    sd.multi_solve_environment()


if __name__ == "__main__":
    main()
