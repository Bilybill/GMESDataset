set -e
python compare_gravity_forward.py \
  --density /path/to/density.npy \
  --in_size 66,41,20 \
  --heights 0,200,400,600,800 \
  --sample 100,100,100 \
  --weight work_dirs/gra_forward/accurate.pth
