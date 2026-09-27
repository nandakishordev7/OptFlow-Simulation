Put one .npz file name per line in a text file here, e.g. splits/kitti_test.txt, then run
    python scripts/evaluate.py --data data/kitti --file_list splits/kitti_test.txt ...
Lines starting with # are ignored. The ".npz" extension is optional.
The official 50-file KITTI test split still needs to be confirmed (NSFP code / OptFlow supplementary).
