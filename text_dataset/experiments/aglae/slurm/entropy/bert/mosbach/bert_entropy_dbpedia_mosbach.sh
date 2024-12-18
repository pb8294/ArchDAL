#!/usr/bin/zsh
#SBATCH --mem=32gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --job-name=glae_dbpedia_entropy_bert_mosbach
#SBATCH --output=/mnt/work/lrauch/logs/aglae/%x_%a.log
#SBATCH --array=1-5%5
date;hostname;pwd
source /mnt/home/lrauch/.zshrc
conda activate uncertainty-evaluation

cd /mnt/home/lrauch/projects/dal-toolbox/experiments/aglae/
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

MODEL=bert
DATASET=dbpedia
STRATEGY=uncertainty

N_INIT=100
ACQ_SIZE=100
N_ACQ=15
GROUP=bert_entropy_dbpedia_mosbach
SEED=$SLURM_ARRAY_TASK_ID
N_EPOCHS=20
LR=2e-5
WARMUP_RATIO=0.1

init_pool_file=~/projects/dal-toolbox/experiments/aglae/initial_pools/dbpedia/random_${N_INIT}_seed${SEED}.json

OUTPUT_DIR=/mnt/work/glae/glae-results/${DATASET}/$MODEL/${STRATEGY}/mosbach/sub/N_INIT${N_INIT}__ACQ_SIZE${ACQ_SIZE}__N_ACQ${N_ACQ}/seed${SLURM_ARRAY_TASK_ID}

echo "Writing results to ${OUTPUT_DIR}"

srun python -u al_txt.py \
    model=$MODEL \
    dataset=$DATASET \
    output_dir=$OUTPUT_DIR \
    random_seed=$SEED \
    al_strategy=$STRATEGY \
    al_cycle.n_init=$N_INIT \
    al_cycle.init_pool_file=$init_pool_file \
    al_cycle.acq_size=$ACQ_SIZE \
    al_cycle.n_acq=$N_ACQ \
    model.n_epochs=$N_EPOCHS \
    model.optimizer.warmup_ratio=$WARMUP_RATIO \
    model.optimizer.lr=$LR \
    wandb.group=$GROUP
    
echo "Finished script."
date