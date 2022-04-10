#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi


# Plotting cifar100 Rebuffi2021Fixing_28_10_cutmix_ddpm models for Linf 1/255
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 1/255 \
    --baseline-train-eps 8/255 \
    --test-eps 1/255 \
    --baseline-model ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --trunk-models ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades1_255ft__20210609_2128/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades1_255.pt \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_autoaugment_trades1_255ft__20210628_1024/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_autoaugment_trades1_255.pt \
        ./models/mrevadv/cifar100/Linf/1_255/mra1_255__Rebuffi2021Fixing_28_10_cutmix_ddpm \
        ./models/mrevadv/cifar100/Linf/1_255/mra1_255_autoaugment__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --branch-model-id Rebuffi2021 \
    --conf-baseline

# Plotting cifar100 Rebuffi2021Fixing_28_10_cutmix_ddpm models for Linf 2/255
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 2/255 \
    --baseline-train-eps 8/255 \
    --test-eps 2/255 \
    --baseline-model ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --trunk-models ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades2_255ft__20210610_2149/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades2_255.pt \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_autoaugment_trades2_255ft__20210629_0210/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_autoaugment_trades2_255.pt \
        ./models/mrevadv/cifar100/Linf/2_255/mra2_255__Rebuffi2021Fixing_28_10_cutmix_ddpm \
        ./models/mrevadv/cifar100/Linf/2_255/mra2_255_autoaugment__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --branch-model-id Rebuffi2021 \
    --conf-baseline

# Plotting cifar100 Rebuffi2021Fixing_28_10_cutmix_ddpm models for Linf 4/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar100 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 4/255 \
    --baseline-train-eps 8/255 \
    --test-eps 4/255 \
    --baseline-model ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --trunk-models ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades4_255ft__20210705_2313/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades4_255.pt \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_autoaugment_trades4_255ft__20210706_1312/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_autoaugment_trades4_255.pt \
        ./models/mrevadv/cifar100/Linf/4_255/mra4_255__Rebuffi2021Fixing_28_10_cutmix_ddpm \
        ./models/mrevadv/cifar100/Linf/4_255/mra4_255_autoaugment__Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --branch-model-id Rebuffi2021 \
    --conf-baseline
