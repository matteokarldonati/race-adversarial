#$ -S /bin/bash

#$ -N ROBERTA_NER
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=50:00:00
#$ -l gpu=true

#$ -cwd

export RACE_DIR=../../../RACE
python3 run_multiple_choice.py \
--task_name race \
--model_name_or_path roberta-base \
--data_dir $RACE_DIR \
--max_seq_length 512 \
--per_gpu_eval_batch_size=4 \
--per_gpu_train_batch_size=2 \
--output_dir ../../output/roberta_base-names \
--do_train \
--learning_rate 1e-5 \
--num_train_epochs 2 \
--gradient_accumulation_steps 16 \
--perturbation_type 'names' \
--perturbation_num_train 2 \
--augment \
--overwrite_output_dir \
