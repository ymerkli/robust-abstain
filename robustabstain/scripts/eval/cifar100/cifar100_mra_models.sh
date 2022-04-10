#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi

# Linf 1/255 mrevadv trained cifar100 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar100/Linf/1_255/mra1_255__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 autoaugment mrevadv trained cifar100 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar100/Linf/1_255/mra1_255_autoaugment__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 2/255 mrevadv trained cifar100 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar100/Linf/2_255/mra2_255__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 autoaugment mrevadv trained cifar100 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar100/Linf/2_255/mra2_255_autoaugment__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 4/255 mrevadv trained cifar100 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar100/Linf/4_255/mra4_255__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 autoaugment mrevadv trained cifar100 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar100/Linf/4_255/mra4_255_autoaugment__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255
