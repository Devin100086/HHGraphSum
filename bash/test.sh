DATASET_NAME="MultiNews" # [MultiNews || CNNDM]
MODEL_NAME="HHDGS" # [HHDGS || HHGS]
TEST_MODEL="multi" # [multi || evalbestFmodel || trainbestmodel || earlystop]


python evaluation.py \
       --cuda \
       --gpu 0 \
       --data_dir data/${DATASET_NAME} \
       --cache_dir cache/${DATASET_NAME} \
       --test_model ${TEST_MODEL} \
       --embedding_path model_hub/Glove/glove.42B.300d.txt \
       --model ${MODEL_NAME} \
       --save_root save/ \
       --log_root log/ \
       -m 9

echo "Test HHGraphSum has completed."