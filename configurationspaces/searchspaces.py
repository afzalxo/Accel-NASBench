import random
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.configuration_space import Configuration


def CustomSearchable(e, k, la, se):
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    blocktypes = ["MB"]
    layer_confs = []
    bchoices = random.choices(blocktypes, k=num_blocks)
    kchoices = k
    echoices = e
    lchoices = la
    sechoices = se
    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
            sechoices[i],
        ]
        layer_confs.append(conf)
    return layer_confs


class EfficientNetSS:
    # Wrapper around ConfigSpace
    def __init__(self):
        try:
            self.ss = cs_json.read(
                open("configurationspaces/effnet_configspace.json").read()
            )
        except FileNotFoundError:
            raise FileNotFoundError("File effnet_configspace.json does not exist...")

    def random_sample(self, num_samples=1):
        return self.ss.sample_configuration(num_samples)

    def size(self):
        return self.ss.estimate_size()

    def __len__(self):
        return self.ss.estimate_size()

    def __str__(self):
        return str(self.ss)

    def efficientnet_b0(self):
        exps = [1, 6, 6, 6, 6, 6, 6]
        kerns = [3, 3, 5, 3, 5, 5, 3]
        lay = [1, 2, 2, 3, 3, 4, 1]
        se = [True for i in range(7)]
        return self.manual_sample([exps, kerns, lay, se])

    def manual_sample(self, sample_config: list):
        default_config_dict = self.ss.get_default_configuration().get_dictionary()
        for k, v in default_config_dict.items():
            block_num = int(k.split("_")[0][-1])
            if "_e" in k:
                default_config_dict[k] = sample_config[0][block_num]
            elif "_k" in k:
                default_config_dict[k] = sample_config[1][block_num]
            elif "_l" in k:
                default_config_dict[k] = sample_config[2][block_num]
            elif "_se" in k:
                default_config_dict[k] = sample_config[3][block_num]
        ret = Configuration(self.ss, values=default_config_dict)
        return ret
