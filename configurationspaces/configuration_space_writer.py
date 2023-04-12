from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
from ConfigSpace.read_and_write import json

cs = ConfigurationSpace(
    seed=1,
    space={
        "block0_e": Categorical('block0_e', [1, 4, 6], default=1),
        "block1_e": Categorical('block1_e', [1, 4, 6], default=1),
        "block2_e": Categorical('block2_e', [1, 4, 6], default=1),
        "block3_e": Categorical('block3_e', [1, 4, 6], default=1),
        "block4_e": Categorical('block4_e', [1, 4, 6], default=1),
        "block5_e": Categorical('block5_e', [1, 4, 6], default=1),
        "block6_e": Categorical('block6_e', [1, 4, 6], default=1),
        "block0_k": Categorical('block0_k', [3, 5], default=3),
        "block1_k": Categorical('block1_k', [3, 5], default=3),
        "block2_k": Categorical('block2_k', [3, 5], default=3),
        "block3_k": Categorical('block3_k', [3, 5], default=3),
        "block4_k": Categorical('block4_k', [3, 5], default=3),
        "block5_k": Categorical('block5_k', [3, 5], default=3),
        "block6_k": Categorical('block6_k', [3, 5], default=3),
        "block0_l": Categorical('block0_l', [1, 2, 3], default=1),
        "block1_l": Categorical('block1_l', [1, 2, 3], default=1),
        "block2_l": Categorical('block2_l', [1, 2, 3], default=1),
        "block3_l": Categorical('block3_l', [1, 2, 3], default=1),
        "block4_l": Categorical('block4_l', [1, 2, 3], default=1),
        "block5_l": Categorical('block5_l', [1, 2, 3, 4], default=1),
        "block6_l": Categorical('block6_l', [1, 2, 3], default=1),
        "block0_se": Categorical('block0_se', [True, False], default=False),
        "block1_se": Categorical('block1_se', [True, False], default=False),
        "block2_se": Categorical('block2_se', [True, False], default=False),
        "block3_se": Categorical('block3_se', [True, False], default=False),
        "block4_se": Categorical('block4_se', [True, False], default=False),
        "block5_se": Categorical('block5_se', [True, False], default=False),
        "block6_se": Categorical('block6_se', [True, False], default=False),
    }
)

with open('configspace.json', 'w') as fh:
    fh.write(json.write(cs))
