import os
import numpy as np

proc_dir = r"d:\Saro Bci And BioMechanics\Saro Bci Project\data\processed"
for root, dirs, files in os.walk(proc_dir):
    for f in files:
        if f.endswith(".npy"):
            path = os.path.join(root, f)
            arr = np.load(path)
            if len(arr.shape) >= 2 and arr.shape[1] > 10000:
                print(f"Trimming {f} from {arr.shape} to (:, 10000)")
                np.save(path, arr[:, :10000])
