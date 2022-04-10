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


# Plotting sbb_l resnet50 models for Linf 1/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 1/255 \
    --baseline-train-eps 8/255 \
    --test-eps 1/255 \
    --baseline-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-models \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades1_255ft__20210625_1140/resnet50_sbb_l_trades1_255.pt \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_stdaug_trades1_255ft__20210625_1409/resnet50_sbb_l_stdaug_trades1_255.pt \
        ./models/mrevadv/sbb_l/Linf/1_255/mra1_255__resnet50_sbb_l_trades8_255 \
        ./models/mrevadv/sbb_l/Linf/1_255/mra1_255_stdaug__resnet50_sbb_l_trades8_255 \
    --branch-model-id ResNet50 \
    --conf-baseline

# Plotting sbb_l resnet50 models for Linf 2/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 2/255 \
    --baseline-train-eps 8/255 \
    --test-eps 2/255 \
    --baseline-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-models \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades2_255ft__20210627_2226/resnet50_sbb_l_trades2_255.pt \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_stdaug_trades2_255ft__20210628_0228/resnet50_sbb_l_stdaug_trades2_255.pt \
        ./models/mrevadv/sbb_l/Linf/2_255/mra2_255__resnet50_sbb_l_trades8_255 \
        ./models/mrevadv/sbb_l/Linf/2_255/mra2_255_stdaug__resnet50_sbb_l_trades8_255 \
    --branch-model-id ResNet50 \
    --conf-baseline

# Plotting sbb_l resnet50 models for Linf 4/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset $DATASET \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 4/255 \
    --baseline-train-eps 8/255 \
    --test-eps 4/255 \
    --baseline-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-models \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades4_255ft__20210710_0058/resnet50_sbb_l_trades4_255.pt \
        ./models/adv/sbb_l/Linf/resnet50_sbb_l_stdaug_trades4_255ft__20210710_0502/resnet50_sbb_l_stdaug_trades4_255.pt \
        ./models/mrevadv/sbb_l/Linf/4_255/mra4_255__resnet50_sbb_l_trades8_255 \
        ./models/mrevadv/sbb_l/Linf/4_255/mra4_255_stdaug__resnet50_sbb_l_trades8_255 \
    --branch-model-id ResNet50 \
    --conf-baseline
