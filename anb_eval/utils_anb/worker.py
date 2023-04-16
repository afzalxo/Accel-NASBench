# Most of the code in this file uses job-submission slurm of our in-house cluster. Hence, it will need to be adopted to run on other clusters. Please file an issue if you need help with this.

import json
import csv
import os
import subprocess
import time
from utils.utils_nasnet import actions_indices_to_config
import wandb


class WorkerSimulatedRS(object):
    def __init__(self):
        self.acc = None
        self.actions_index = None

    def Query(self, acc_surrogate, actions_index):
        self.actions_index = actions_index
        self.acc = acc_surrogate.query(actions_index)[0]
        print(f"Accuracy: {self.acc}")


class WorkerSimulatedPG(object):
    def __init__(self, arch_ep, episode, configspace, controller_inst):
        self.acc = None
        self.configspace = configspace
        self.arch_ep, self.episode = arch_ep, episode

        self.actions_p, self.actions_log_p, self.actions_index = controller_inst.sample()
        self.actions_index = self.actions_index.cpu()
        self.actions_config, self.design = actions_indices_to_config(self.actions_index, self.configspace)

    def Query(self, acc_surrogate):
        self.acc = acc_surrogate.query(self.actions_config)[0]
        print(f"Arch Ep: {self.arch_ep}, Episode: {self.episode}, Accuracy: {self.acc}")


class WorkerSimulatedPGMO(object):
    def __init__(self, arch_ep, episodes, configspace, controller_inst):
        self.acc = []
        self.biobj_metric = []
        self.actions_indices = []
        self.actions_configs = []
        self.designs = []
        self.configspace = configspace
        self.arch_ep, self.episodes = arch_ep, episodes
        for i in range(episodes):
            actions_p, actions_log_p, actions_index = controller_inst.sample()
            actions_index = actions_index.cpu()
            actions_config, design = actions_indices_to_config(actions_index, self.configspace)
            self.actions_indices.append(actions_index)
            self.actions_configs.append(actions_config)
            self.designs.append(design)

    def Query(self, acc_surrogate, biobj_surrogate):
        self.acc = list(acc_surrogate.query_batch(self.actions_configs)) # [0]
        self.biobj_metric = list(biobj_surrogate.query_batch(self.actions_configs)) # [0]
        print(f"Arch Ep: {self.arch_ep}, Episodes: {self.episodes}, Accuracy: {self.acc}, Second Objective: {self.biobj_metric}")
        return self.acc, self.biobj_metric


class WorkerTACC(object):
    def __init__(self, search_algo, arch_ep, episode, configspace, controller_inst):
        self.acc = None
        self.launch_time = None
        self.username = None
        self.ssh_key = None
        self.job_id = None
        self.search_algo = search_algo
        self.configspace = configspace
        self.controller_inst = controller_inst
        self.actions_p, self.actions_log_p, self.actions_index = None, None, None
        self.actions_config, self.design = None, None
        self.episode = episode
        self.arch_ep = arch_ep
        self.json_path_tuxiv = f"rl_search_jsons/sample_{arch_ep}_{episode}.json"
        self.json_path = os.path.join(
            "/home/aahmadaa/NASBenchFPGA/dataset_collection/rl_search_jsons",
            f"sample_{arch_ep}_{episode}.json",
        )
        self.launch_time = time.time()
        self.trynum = 162

    def SampleController(self):
        self.actions_p, self.actions_log_p, self.actions_index = self.controller_inst.sample()
        self.actions_index = self.actions_index.cpu()
        self.actions_config, self.design = actions_indices_to_config(self.actions_index, self.configspace)

    def SaveSampleJSON(self):
        self.SampleController()
        with open(self.json_path, 'w') as fh:
            json.dump(self.actions_config.get_dictionary(), fh)

    def LaunchTrainJob(self, lock, version=0):
        usernames = ["aahmadaa1", "aahmadaa", "hfanah"]
        ssh_keys = [
            "/home/aahmadaa/ssh-keys/key_mac",
            "/home/aahmadaa/ssh-keys/key_mac",
            "/home/aahmadaa/ssh-keys/id_rsa_fan",
        ]
        userindex = self.episode % len(usernames)
        username = usernames[userindex]
        self.username = username
        self.ssh_key = ssh_key = ssh_keys[userindex]
        lock.acquire()
        print('Lock Acquired... Preparing to Submit Job')
        self.SaveSampleJSON()
        os.system(f"tacc config -u {username}")
        os.system(f"tacc config -f {ssh_key}")
        if self.job_id is not None:
            os.system(f'tacc cancel -j {self.job_id}')
        dir_back = os.getcwd()
        os.chdir("/home/aahmadaa/NASBenchFPGA/dataset_collection")
        os.system("rm -rf /home/aahmadaa/NASBenchFPGA/dataset_collection/logs/*")
        write_new_tuxiv(
            self.arch_ep, self.episode, version, self.json_path_tuxiv, self.search_algo
        )
        print(
            f"[SUBMIT-JOB][START] Epoch {self.arch_ep}, Episode {self.episode} to TACC @ {username}..."
        )
        #os.system("tacc submit")
        result = subprocess.run(['tacc', 'submit'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = result.stdout.decode()
        last_line = result.splitlines()[-1]
        self.job_id = int(last_line.split(' ')[-1])
        print(
            f"[SUBMIT-JOB][FINISH] Epoch {self.arch_ep}, Episode {self.episode} Job ID: {self.job_id}..."
        )
        os.chdir(dir_back)
        # self.TACCWaitJobRun()
        time.sleep(2)
        lock.release()
        self.launch_time = time.time()

    def PollWBTrainability(self, lock, version):
        api = wandb.Api()
        trainable = False
        finished = False
        while not finished:
            try:
                artifact_address = f"europa1610/NASBenchFPGA/trainability-search-try{self.trynum}-{self.search_algo}-{self.arch_ep}-{self.episode}-{version}:v0"
                artifact = api.artifact(artifact_address, type='custom')
                trainable = bool(artifact.metadata['trainable'])
                print(f"[RECV-ART][Trainability] established Job ID {self.job_id} Epoch {self.arch_ep}, Episode {self.episode}, Username {self.username}, trainable: {trainable}... ")
                finished = True
            except KeyboardInterrupt:
                print('Aborting...')
                if self.job_id is not None:
                    lock.acquire()
                    os.system(f"tacc config -u {self.username}")
                    os.system(f"tacc config -f {self.ssh_key}")
                    os.system(f'tacc cancel -j {self.job_id}')
                    lock.release()
                exit(0)
            except:
                passed_since_launch = (time.time() - self.launch_time) / 60
                print(
                    f"[WAIT-ART][Trainability], Job ID {self.job_id}, Arch Ep {self.arch_ep}, Episode {self.episode}, Username {self.username}, Waiting for artifact since {passed_since_launch:.2f} mins"
                )
                time.sleep(10)
                if passed_since_launch > 5:
                    print('[WAIT-ART][Trainability][TIMEOUT] Job ID {self.job_id} timed out waiting for trainability response from W&B, Resubmitting Job...')
                    if self.job_id is not None:
                        lock.acquire()
                        os.system(f"tacc config -u {self.username}")
                        os.system(f"tacc config -f {self.ssh_key}")
                        os.system(f'tacc cancel -j {self.job_id}')
                        lock.release()
                    trainable = False
                    finished = True
        return trainable

    def PollWBArtifact(self, lock):
        version = 0
        while not self.PollWBTrainability(lock, version):
            version += 1
            self.LaunchTrainJob(lock, version)
        api = wandb.Api()
        finished = False
        while not finished:
            try:
                artifact_address = f"europa1610/NASBenchFPGA/models-search-try{self.trynum}-{self.search_algo}-{self.arch_ep}-{self.episode}:v0"
                artifact = api.artifact(artifact_address, type='model')
                md = artifact.metadata["model_metadata"]
                self.acc = float(md["best_acc_top1"])
                time_took = (time.time() - self.launch_time) / 60
                print(
                    f"[RECV-ART][ACC] Artifact found, Job ID: {self.job_id}, Epoch {self.arch_ep}, Episode {self.episode}, Username {self.username}, Acc: {self.acc}... Worker finished in {time_took:.2f} mins..."
                )
                finished = True
            except KeyboardInterrupt:
                print("Aborting...")
                if self.job_id is not None:
                    os.system(f'tacc cancel -j {self.job_id}')
                exit(0)
            except:
                passed_since_launch = (time.time() - self.launch_time) / 60
                print(
                    f"[WAIT-ART][ACC] Job ID: {self.job_id}, Epoch {self.arch_ep}, Episode {self.episode}, Username {self.username}, Time since worker start: {passed_since_launch:.2f} mins"
                )
                time.sleep(20)

    def TACCWaitJobRun(self):
        running = False
        job_state = None
        while not running:
            result = subprocess.run(['tacc', 'ps'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = result.stdout.decode()
            result = result.splitlines()[1:]
            for res in result:
                strip_res = [x for x in res.split(' ') if x.strip()]
                if int(strip_res[0]) == self.job_id:
                    job_state = strip_res[4]
            if job_state == 'R':
                running = True
            else:
                time.sleep(2)
        print(f'Job {self.job_id} running...')
        return


def write_new_tuxiv(arch_ep, episode, version, json_path, search_algo):
    with open('/home/aahmadaa/NASBenchFPGA/nas_optimizers/AlgoUtils/nodelist.csv', 'r') as fh:
        reader = csv.reader(fh)
        for line in reader:
            nodelist = line
    # nodelist = ['10-0-7-19', '10-0-8-18', '10-0-8-19', '10-0-4-[10-11]', '10-0-4-[12-13]', '10-0-6-[10-11]', '10-0-6-[12-13]']
    if '[' in nodelist[episode]:
        ngpus = 2
        nnodes = 2
    else:
        ngpus = 4
        nnodes = 1
    new_command = f'   - CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 ${{TACC_WORKDIR}}/train_design_proxified.py --cfg_path ./configs/conf_tacc.cfg --arch_epoch {arch_ep} --episode {episode} --architecture_json {json_path} --search_algo {search_algo} --version {version}\n'
    with open("tuxiv.conf", "r") as f:
        lines = f.readlines()

    for i, l in enumerate(lines):
        if "CUDA_VISIBLE_DEVICES" in l:
            lines[i] = new_command
        if 'nodes=' in l:
            lines[i] = f'      - nodes={nnodes}\n'
        if 'ntasks-per-node' in l:
            lines[i] = f'      - ntasks-per-node={ngpus}\n'
        if 'gres' in l:
            lines[i] = f'      - gres=gpu:{ngpus}\n'
        if 'nodelist' in l:
            lines[i] = f'      - nodelist={nodelist[episode]}\n'


    # Write the modified lines back to the file
    with open("tuxiv.conf", "w") as f:
        f.writelines(lines)
