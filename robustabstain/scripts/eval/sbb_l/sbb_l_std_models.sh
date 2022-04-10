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


# standard trained sbb_l models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/std/sbb_l \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255 2/255 4/255 8/255 16/255 \
    --use-exist-log

# standard trained sbb_l models evaluated for L2 robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/std/sbb_l \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm L2 \
    --test-eps 0.25 0.5 1.0 1.5 2.0 \
    --use-exist-log
