import pickle
import numpy as np

def inspect_init_state(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"\n📂 File: {path}")
    for k, v in data.items():
        print(f"  {k:>15}: {type(v)} shape={np.shape(v)}")

inspect_init_state("./mpc_init/init_000.pkl")
inspect_init_state("./mpc_target_U/target_000.pkl")