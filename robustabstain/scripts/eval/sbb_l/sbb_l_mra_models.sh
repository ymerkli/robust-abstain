#!/bin/bash

DATASET=$1
EVAL_SET=$2

if [ -z "$DATASET" ]
then
    DATASET="sbb_l"
fi

if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi


# Linf 1/255 mrevadv trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/sbb_l/Linf/1_255/mra1_255__resnet50_sbb_l_trades8_255 \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255
#    --use-exist-log

# Linf 1/255 stdaug mrevadv trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/sbb_l/Linf/1_255/mra1_255_stdaug__resnet50_sbb_l_trades8_255 \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255
#    --use-exist-log

# Linf 2/255 mrevadv trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/sbb_l/Linf/2_255/mra2_255__resnet50_sbb_l_trades8_255 \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255
#    --use-exist-log

# Linf 2/255 stdaug mrevadv trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/sbb_l/Linf/2_255/mra2_255_stdaug__resnet50_sbb_l_trades8_255 \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255
#    --use-exist-log

# Linf 4/255 mrevadv trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/sbb_l/Linf/4_255/mra4_255__resnet50_sbb_l_trades8_255 \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255
#    --use-exist-log

# Linf 4/255 stdaug mrevadv trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/sbb_l/Linf/4_255/mra4_255_stdaug__resnet50_sbb_l_trades8_255 \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255
#    --use-exist-log
