#$ -S /bin/bash

#$ -N BERT_BASE_ORG
#$ -j y

#$ -l tmem=16G
#$ -l h_rt=50:00:00
#$ -l gpu=true

#$ -cwd

export PATH=/share/apps/python-3.7.2-shared/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/python-3.7.2-shared/lib:${LD_LIBRARY_PATH}

source /share/apps/examples/source_files/cuda/cuda-10.1.source

export RACE_DIR=../../../RACE
python3 ../../bert-race-data-augmentation/run_race.py \
--data_dir=$RACE_DIR \
--bert_model=../../../models/bert-base-uncased \
--output_dir=../../output/bert_base_org \
--max_seq_length=380 \
--do_train \
--do_eval \
--do_lower_case \
--train_batch_size=32 \
--eval_batch_size=4 \
--learning_rate=5e-5 \
--num_train_epochs=2 \
--gradient_accumulation_steps=8 \
--perturbation_type='ORG' \
--perturbation_num=2 \
--augment \
--save_total_limit 1 \
