#!/bin/bash

export YAML_NAME=oneshot_oBERT_nm
export RECIPE=integrations/huggingface-transformers/recipes/${YAML_NAME}.yaml

CUDA_VISIBLE_DEVICES=0 python src/sparseml/transformers/question_answering.py \
--model_name_or_path neuralmagic/oBERT-teacher-squadv1 \
--dataset_name squad \
--do_train \
--fp16 \
--do_eval \
--do_oneshot \
--per_device_eval_batch_size 16 \
--max_seq_length 384 \
--doc_stride 128 \
--preprocessing_num_workers 8 \
--seed 42 \
--recipe ${RECIPE} \
--output_dir integrations/huggingface-transformers/output_dir/test_obert1116_debug \
--overwrite_output_dir \
--skip_memory_metrics true \
--report_to none