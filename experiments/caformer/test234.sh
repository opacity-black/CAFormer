
SCRIPT=caformer
CONFIG=vit_cte_cma_st_f3_lr5e5
nohup python tracking/test.py \
    ${SCRIPT} \
    ${CONFIG} \
    --dataset RGBT234 \
    --threads 3 \
    --num_gpus 1 \
    --vis_gpus 3 \
    >./experiments/${SCRIPT}/test_${SCRIPT}-${CONFIG}.log 2>&1 &

