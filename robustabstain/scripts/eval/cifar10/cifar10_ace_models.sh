#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi

# COLT trained models
python3 ./eval/run_ace.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --model ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --train-batch 20 \
    --test-batch 20 \
    --cert-domain zono \
    --adv-norm Linf \
    --test-eps 1/255 2/255 \
    --use-exist-log


# IBP trained models
python3 ./eval/run_ace.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --model ./models/ace/cifar10/Linf/ACE_Net_IBP_cert_2_255/C3_ACE_Net_IBP_cert_cifar10_2_255.pt \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --train-batch 20 \
    --test-batch 20 \
    --cert-domain box \
    --adv-norm Linf \
    --test-eps 1/255 2/255 \
    --use-exist-log

python3 ./eval/run_ace.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --model ./models/ace/cifar10/Linf/ACE_Net_IBP_cert_8_255/C3_ACE_Net_IBP_cert_cifar10_8_255.pt \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --train-batch 20 \
    --test-batch 20 \
    --cert-domain box \
    --adv-norm Linf \
    --test-eps 4/255 8/255 \
    --use-exist-log
