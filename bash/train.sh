DATASET_NAME="MultiNews" # [MultiNews || CNNDM]
MODEL_NAME="HHDGS" # [HHDGS || HHGS]

python train.py \
       --cuda \
       --gpu 0 \
       --data_dir data/${DATASET_NAME} \
       --cache_dir cache/${DATASET_NAME} \
       --embedding_path model_hub/Glove/glove.42B.300d.txt \
       --model ${MODEL_NAME} \
       --save_root save/ \
       --log_root log/ \
       --lr_descent \
       --grad_clip \
       -m 9

echo "Training HHGraphSum has completed."