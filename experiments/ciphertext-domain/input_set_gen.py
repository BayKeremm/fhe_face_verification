import numpy as np

def generate_inputset(size, max_val_int, vector_dim, min_val=0):
    return [(
        [np.random.randint(min_val, max_val_int + 1) for _ in range(vector_dim)],
        [np.random.randint(min_val, max_val_int + 1) for _ in range(vector_dim)]
    ) for _ in range(size)]

# Generate inputsets and store in a dictionary
inputsets = {}
for vector_dim in range(40, 45):
    inputsets[f"{vector_dim}_bit4"] = generate_inputset(size=10, max_val_int=15, vector_dim=vector_dim)
    inputsets[f"{vector_dim}_bit5"] = generate_inputset(size=10, max_val_int=31, vector_dim=vector_dim)
    inputsets[f"{vector_dim}_bit4_signed"] = generate_inputset(size=10, max_val_int=7, min_val=-8, vector_dim=vector_dim)
    inputsets[f"{vector_dim}_bit5_signed"] = generate_inputset(size=10, max_val_int=15, min_val=-16, vector_dim=vector_dim)


inputsets["128_bit4"] =  generate_inputset(size=10, max_val_int=15, min_val=0, vector_dim=128)
inputsets["128_bit4_signed"] =  generate_inputset(size=10, max_val_int=7, min_val=-8, vector_dim=128)
inputsets["128_bit5"] =  generate_inputset(size=10, max_val_int=15, min_val=-16, vector_dim=128)
inputsets["128_bit5_signed"] =  generate_inputset(size=10, max_val_int=15, min_val=-16, vector_dim=128)
# Save to a .npz file
npz_path = "./inputsets.npz"
np.savez_compressed(npz_path, **inputsets)
