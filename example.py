import accelnb as anb
from configurationspaces.searchspaces import EfficientNetSS as ss

seed = 3
# Create ensemble instance
ensemble_inst_acc = anb.ANBEnsemble("xgb", seed=seed)
ensemble_inst_thr = anb.ANBEnsemble(
    "xgb", device="tpuv2", metric="throughput", seed=seed
)

# Load ensemble
acc_model = ensemble_inst_acc.load_ensemble()
thr_model = ensemble_inst_thr.load_ensemble()

# Create search space instance
search_space = ss()

# Obtain random sample from configspace
test_sample_rand = search_space.random_sample(1)

# Or use manual_sample to specify configuration:
test_sample_man = search_space.manual_sample(
    [
        [1, 6, 6, 6, 6, 6, 6],  # Expansion Factor for the 7 blocks
        [3, 3, 5, 3, 5, 5, 3],  # Kernel Sizes
        [1, 2, 2, 3, 3, 4, 1],  # Number of Layers in block
        [True, True, True, True, True, True, True],   # Squeeze-Excite state
    ]
)

# Pass a list of samples as input to query to get a list of acc/thr values
mean_acc, std = acc_model.query([test_sample_rand, test_sample_man])
print(f"Mean Accuracy: {mean_acc}\nStd Acc: {std}")
mean_thr, std = thr_model.query([test_sample_rand, test_sample_man])
print(f"Mean Throughput: {mean_thr}\nStd Thr: {std}")
