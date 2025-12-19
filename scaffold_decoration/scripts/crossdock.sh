CUDA_VISIBLE_DEVICES=2 python run.py inference.crossdock \
    --data_root /home/minyeong/ACE_Deploy/data/crossdock/raw \
    --save_dir result_inference/crossdock_1.3_bump \
    --inverse_temperature 1.3 \
    --use_bump
