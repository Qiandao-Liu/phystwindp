# count_gs_points.py
import pickle

def count_cloth_points(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    if 'object_points' in data:
        first_frame = data['object_points'][0]  # list of frames
    else:
        raise ValueError(f"❌ No 'object_points' found. Found keys: {list(data.keys())}")

    num_points = first_frame.shape[0]
    print(f"✅ Cloth GS points: {num_points}")

if __name__ == "__main__":
    count_cloth_points("data/different_types/double_lift_cloth_1/final_data.pkl")
