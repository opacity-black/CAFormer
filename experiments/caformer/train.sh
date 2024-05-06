SCRIPT=caformer
CONFIG=vit_cte_cma_st_f3_lr5e5
nohup python tracking/train.py \
    --script ${SCRIPT} \
    --config ${CONFIG} \
    --save_dir ./output \
    --mode multiple \
    --nproc_per_node 2 \
    --use_wandb 0 \
    --use_lmdb 0 \
    --vis_gpus 4,5 \
    >./experiments/${SCRIPT}/train_log/${SCRIPT}-${CONFIG}.log 2>&1 &