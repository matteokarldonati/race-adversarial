#$ -S /bin/bash

#$ -N ner_evaluation
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true,gpu_p100=yes

#$ -cwd

export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}

source /share/apps/examples/source_files/cuda/cuda-10.1.source

export RACE_DIR=../../RACE

# BERT_base

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_large-race \
--data_dir $RACE_DIR \
--max_seq_length 320 \
--per_gpu_eval_batch_size=1 \
--output_dir ../output/bert_base_output_names \
--perturbation_type 'names' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_large-race \
--data_dir $RACE_DIR \
--max_seq_length 320 \
--per_gpu_eval_batch_size=1 \
--output_dir ../output/bert_base_output_org \
--perturbation_type 'ORG' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_large-race \
--data_dir $RACE_DIR \
--max_seq_length 320 \
--per_gpu_eval_batch_size=1 \
--output_dir ../output/bert_base_output_gpe \
--perturbation_type 'GPE' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_large-race \
--data_dir $RACE_DIR \
--max_seq_length 320 \
--per_gpu_eval_batch_size=1 \
--output_dir ../output/bert_base_output_loc \
--perturbation_type 'LOC' \
--perturbation_num_test 25 \
--augment \

python3 ../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../models/bert_large-race \
--data_dir $RACE_DIR \
--max_seq_length 320 \
--per_gpu_eval_batch_size=1 \
--output_dir ../output/bert_base_output_norp \
--perturbation_type 'NORP' \
--perturbation_num_test 25 \
--augment \
