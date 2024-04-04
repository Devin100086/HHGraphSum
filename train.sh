python train.py \
       --cuda \
       --gpu 0 \
       --data_dir data/CNNDM \
       --cache_dir cache/CNNDM \
       --embedding_path model_hub/Glove/glove.42B.300d.txt \
       --model HHDGS \
       --save_root save/ \
       --log_root log/ \
       --lr_descent \
       --grad_clip \
       -m 3
echo "Training HHGraphSum has completed."