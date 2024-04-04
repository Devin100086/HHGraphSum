python train.py \
       --cuda \
       --gpu 0 \
       --data_dir data/MultiNews \
       --cache_dir cache/MultiNews \
       --embedding_path model_hub/Glove/glove.42B.300d.txt \
       --model HHDGS \
       --save_root save/ \
       --log_root log/ \
       --lr_descent \
       --grad_clip \
       -m 9
echo "Training HHGraphSum has completed."