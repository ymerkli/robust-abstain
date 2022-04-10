#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi

# Linf 1/255 mrevadv trained mtsd_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/mtsd_l/Linf/1_255/mra1_255__resnet50_mtsd_l_trades8_255 \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 stdaug mrevadv trained mtsd_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/mtsd_l/Linf/1_255/mra1_255_stdaug__resnet50_mtsd_l_trades8_255 \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 2/255 mrevadv trained mtsd_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/mtsd_l/Linf/2_255/mra2_255__resnet50_mtsd_l_trades8_255 \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 stdaug mrevadv trained mtsd_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/mtsd_l/Linf/2_255/mra2_255_stdaug__resnet50_mtsd_l_trades8_255 \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 4/255 mrevadv trained mtsd_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/mtsd_l/Linf/4_255/mra4_255__resnet50_mtsd_l_trades8_255 \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 stdaug mrevadv trained mtsd_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/mtsd_l/Linf/4_255/mra4_255_stdaug__resnet50_mtsd_l_trades8_255 \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255
