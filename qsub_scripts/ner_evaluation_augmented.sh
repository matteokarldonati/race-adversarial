#$ -S /bin/bash

#$ -N ner_evaluation_augmented
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true

#$ -cwd

export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}

source /share/apps/examples/source_files/cuda/cuda-10.1.source

export RACE_DIR=../../RACE

# BERT_base

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_base-name \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/bert_base_names_output_names \
--perturbation_type 'names' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_base-org \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/bert_base_org_output_org \
--perturbation_type 'ORG' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_base-gpe \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/bert_base_gpe_output_gpe \
--perturbation_type 'GPE' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_base-loc \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/bert_base_loc_output_loc \
--perturbation_type 'LOC' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_base-norp \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/bert_base_norp_output_norp \
--perturbation_type 'NORP' \
--perturbation_num_test 25 \
--augment \

# ALBERT_base

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/albert_base-names \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/albert_base_names_output_names \
--perturbation_type 'names' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/albert_base-org \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/albert_base_org_output_org \
--perturbation_type 'ORG' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/albert_base-gpe \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/albert_base_gpe_output_gpe \
--perturbation_type 'GPE' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/albert_base-loc \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/albert_base_loc_output_loc \
--perturbation_type 'LOC' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/albert_base-norp \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/albert_base_norp_output_norp \
--perturbation_type 'NORP' \
--perturbation_num_test 25 \
--augment \

# RoBERTa_base

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/roberta_base-names \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/roberta_base_names_output_names \
--perturbation_type 'names' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/roberta_base-org \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/roberta_base_org_output_org \
--perturbation_type 'ORG' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/roberta_base-gpe \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/roberta_base_gpe_output_gpe \
--perturbation_type 'GPE' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/roberta_base-loc \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/roberta_base_loc_output_loc \
--perturbation_type 'LOC' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/roberta_base-norp \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/roberta_base_norp_output_norp \
--perturbation_type 'NORP' \
--perturbation_num_test 25 \
--augment \

# TinyBERT

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/tinybert-names \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/tinybert_base_names_output_names \
--perturbation_type 'names' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/tinybert-org \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/tinybert_base_org_output_org \
--perturbation_type 'ORG' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/tinybert-gpe \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/tinybert_base_gpe_output_gpe \
--perturbation_type 'GPE' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/tinybert-loc \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/tinybert_base_loc_output_loc \
--perturbation_type 'LOC' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/tinybert-norp \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--output_dir ../output/tinybert_base_norp_output_norp \
--perturbation_type 'NORP' \
--perturbation_num_test 25 \
--augment \
