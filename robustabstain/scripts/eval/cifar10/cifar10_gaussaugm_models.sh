#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi

# noise augmentation trained cifar10 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/augm/cifar10 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255 2/255 4/255 8/255 16/255

# noise augmentation trained cifar10 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/augm/cifar10 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm L2 \
    --test-eps 0.25 0.5 1.0 1.5 2.0
