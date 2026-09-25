"""Download the preprocessed scene-flow datasets (from the Neural Scene Flow Prior authors).
Works on Windows, Mac, Linux and Colab.

    python scripts/download_data.py kitti
    python scripts/download_data.py nuscenes
"""
import os
import sys
import glob
import shutil
import subprocess
import tarfile
import zipfile

IDS = {
    "kitti": "1pjShY0RxHp0EjkelWGLO_BMR7qCB528p",         # 266 MB
    "argoverse": "1qyTaLz1_CTF3IB1gr3XpIiIDh6klQOA4",     # 370 MB
    "nuscenes": "1mCjDqJzaMdW0iiM2N2J5BNvo04dAvTbx",      # 73 MB
    "flyingthings": "1v9M0sRCHKrPj5phHxC03-WsdhctUL9j9",  # 948 MB
}

name = sys.argv[1] if len(sys.argv) > 1 else "kitti"
if name not in IDS:
    sys.exit(f"unknown dataset '{name}', choose from {list(IDS)}")

try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    import gdown

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
out_dir = os.path.join(root, name)
archive = os.path.join(root, f"{name}.download")
os.makedirs(out_dir, exist_ok=True)

print(f"downloading {name} ...")
gdown.download(id=IDS[name], output=archive, quiet=False)

print("extracting ...")
if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)
elif tarfile.is_tarfile(archive):
    with tarfile.open(archive) as t:
        t.extractall(out_dir)
else:  # not an archive: keep the file as it is
    shutil.move(archive, os.path.join(out_dir, name))
if os.path.exists(archive):
    os.remove(archive)

n = len(glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True))
print(f"done: {n} .npz files in {out_dir}")
