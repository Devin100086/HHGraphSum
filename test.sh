python evaluation.py \
       --cuda \
       --gpu 0 \
       --data_dir data/MultiNews \
       --cache_dir cache/MultiNews \
       --test_model evalbestFmodel \
       --embedding_path model_hub/Glove/glove.42B.300d.txt \
       --model HHDGS \
       --save_root save/ \
       --log_root log/ \
       -m 9
echo "Test HHGraphSum has completed."