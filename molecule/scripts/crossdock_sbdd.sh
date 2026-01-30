CUDA_VISIBLE_DEVICES=2 python run.py inference.crossdock \
    --data_root /home/minyeong/ACE_Deploy/data/crossdock/raw_sbdd \
    --save_dir result_inference/crossdock_sbdd_1.3_bump \
    --inverse_temperature 1.3 \
    --use_bump
