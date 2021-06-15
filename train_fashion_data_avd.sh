CUDA_VISIBLE_DEVICES=0,1,2 python run.py \
          --checkpoint '/Users/john.lim/Downloads/00000099-cpk-reconstruction.pth' \
          --config 'config/fashion.yaml' \
          --device_ids 0,1,2 \
          --mode train_avd
