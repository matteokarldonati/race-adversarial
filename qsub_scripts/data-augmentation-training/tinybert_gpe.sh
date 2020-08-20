#$ -S /bin/bash

#$ -N TINYBERT_GPE
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=24:00:00
#$ -l gpu=true

#$ -cwd

export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}

source /share/apps/examples/source_files/cuda/cuda-10.1.source

export RACE_DIR=../../../RACE
python3 ../../run_multiple_choice.py \
--task_name race \
--model_name_or_path ../../../models/tinybert \
--data_dir $RACE_DIR \
--max_seq_length 380 \
--per_gpu_eval_batch_size=4 \
--per_gpu_train_batch_size=4 \
--output_dir ../../output/tinybert-gpe \
--do_train \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--gradient_accumulation_steps 2 \
--perturbation_type 'GPE' \
--perturbation_num_train 3 \
--overwrite_output_dir \
--save_total_limit 1 \
