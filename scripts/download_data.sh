#!/usr/bin/env bash
# Preprocessed datasets released by the Neural Scene Flow Prior authors
# (https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior). Usage: bash scripts/download_data.sh kitti
set -e
pip install -q gdown
declare -A IDS=(
  [kitti]=1pjShY0RxHp0EjkelWGLO_BMR7qCB528p          # 266 MB
  [argoverse]=1qyTaLz1_CTF3IB1gr3XpIiIDh6klQOA4      # 370 MB
  [nuscenes]=1mCjDqJzaMdW0iiM2N2J5BNvo04dAvTbx       # 73 MB
  [flyingthings]=1v9M0sRCHKrPj5phHxC03-WsdhctUL9j9   # 948 MB
)
name=${1:-kitti}
mkdir -p data/$name
gdown "${IDS[$name]}" -O data/$name.download
# the file may be a zip or a tar archive
if unzip -tq data/$name.download >/dev/null 2>&1; then
  unzip -q data/$name.download -d data/$name
else
  tar -xf data/$name.download -C data/$name
fi
rm data/$name.download
echo "npz files found: $(find data/$name -name '*.npz' | wc -l)"
