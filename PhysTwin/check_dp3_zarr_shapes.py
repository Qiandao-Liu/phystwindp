# /workspace/PhysTwin/check_dp3_zarr_shapes.py

import zarr
import numpy as np

zarr_path = "PhysTwin/dp3_data/cloth_aloha_dataset.zarr"
ds = zarr.open(zarr_path, mode='r')

fields = ['point_cloud', 'agent_pos', 'action']
print(f"Checking dataset at: {zarr_path}\n")

for field in fields:
    data = ds[field]
    print(f"[{field}] shape: {data.shape}")
    print(f"  dtype: {data.dtype}")
    if data.shape[0] > 0:
        example = data[0]
        print(f"  example[0] shape: {example.shape}")
    print()
