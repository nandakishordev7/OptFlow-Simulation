"""Print the keys/shapes of the first .npz so you can confirm the loader matches.
python scripts/check_data.py data/kitti"""
import sys, glob, numpy as np
files = sorted(glob.glob(f"{sys.argv[1]}/**/*.npz", recursive=True))
print(len(files), "files")
d = np.load(files[0])
for k in d.files:
    print(f"  {k:12s} {d[k].shape} {d[k].dtype}")
