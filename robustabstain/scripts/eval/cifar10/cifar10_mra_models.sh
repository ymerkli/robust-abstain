#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi

# Linf 1/255 mrevadv trained cifar10 Carmon2019Unlabeled models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/1_255/mra1_255__Carmon2019Unlabeled \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 autoaugment mrevadv trained cifar10 Carmon2019Unlabeled models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/1_255/mra1_255_autoaugment__Carmon2019Unlabeled \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/1_255/mra1_255__Gowal2020Uncovering_28_10_extra \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 autoaugment mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/1_255/mra1_255_autoaugment__Gowal2020Uncovering_28_10_extra \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 mrevadv trained cifar10 resnet50 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/1_255/mra1_255__resnet50_cifar10_trades1_255 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255

# Linf 1/255 autoaugment mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/1_255/mra1_255_autoaugment__resnet50_cifar10_trades1_255 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255


# Linf 2/255 mrevadv trained cifar10 Carmon2019Unlabeled models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/2_255/mra2_255__Carmon2019Unlabeled \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 autoaugment mrevadv trained cifar10 Carmon2019Unlabeled models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/2_255/mra2_255_autoaugment__Carmon2019Unlabeled \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/2_255/mra2_255__Gowal2020Uncovering_28_10_extra \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 autoaugment mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/2_255/mra2_255_autoaugment__Gowal2020Uncovering_28_10_extra \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 mrevadv trained cifar10 resnet50 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/2_255/mra2_255__resnet50_cifar10_trades2_255 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255

# Linf 2/255 autoaugment mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/2_255/mra2_255_autoaugment__resnet50_cifar10_trades2_255 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 2/255


# Linf 4/255 mrevadv trained cifar10 Carmon2019Unlabeled models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/4_255/mra4_255__Carmon2019Unlabeled \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 autoaugment mrevadv trained cifar10 Carmon2019Unlabeled models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/4_255/mra4_255_autoaugment__Carmon2019Unlabeled \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/4_255/mra4_255__Gowal2020Uncovering_28_10_extra \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 autoaugment mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/4_255/mra4_255_autoaugment__Gowal2020Uncovering_28_10_extra \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 mrevadv trained cifar10 resnet50 models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/4_255/mra4_255__resnet50_cifar10_trades4_255 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255

# Linf 4/255 autoaugment mrevadv trained cifar10 Gowal2020Uncovering_28_10_extra models evaluated for Linf robustness
python3 ./eval/eval_models.py \
    --eval-dir ./models/mrevadv/cifar10/Linf/4_255/mra4_255_autoaugment__resnet50_cifar10_trades4_255 \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --evals nat adv \
    --adv-attack apgd \
    --adv-norm Linf \
    --test-eps 4/255
