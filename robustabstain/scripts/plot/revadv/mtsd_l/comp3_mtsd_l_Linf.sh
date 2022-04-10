#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi


# Plotting mtsd_l resnet50 models for Linf 1/255
python3 ./analysis/plotting/plot_revadv.py \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 1/255 \
    --baseline-train-eps 8/255 \
    --test-eps 1/255 \
    --baseline-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --trunk-models \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades1_255ft__20210525_1104/resnet50_mtsd_l_trades1_255.pt \
        ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-models \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades1_255ft__20210525_1104/resnet50_mtsd_l_trades1_255.pt \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_stdaug_trades1_255ft__20210725_1855/resnet50_mtsd_l_stdaug_trades1_255.pt \
        ./models/mrevadv/mtsd_l/Linf/1_255/mra1_255__resnet50_mtsd_l_trades8_255 \
        ./models/mrevadv/mtsd_l/Linf/1_255/mra1_255_stdaug__resnet50_mtsd_l_trades8_255 \
    --branch-model-id ResNet50 \
    --only-comp

# Plotting mtsd_l resnet50 models for Linf 2/255
python3 ./analysis/plotting/plot_revadv.py \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 2/255 \
    --baseline-train-eps 8/255 \
    --test-eps 2/255 \
    --baseline-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --trunk-models \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades2_255ft__20210526_1406/resnet50_mtsd_l_trades2_255.pt \
        ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-models \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades2_255ft__20210526_1406/resnet50_mtsd_l_trades2_255.pt \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_stdaug_trades2_255ft__20210726_0355/resnet50_mtsd_l_stdaug_trades2_255.pt \
        ./models/mrevadv/mtsd_l/Linf/2_255/mra2_255__resnet50_mtsd_l_trades8_255 \
        ./models/mrevadv/mtsd_l/Linf/2_255/mra2_255_stdaug__resnet50_mtsd_l_trades8_255 \
    --branch-model-id ResNet50 \
    --only-comp


# Plotting mtsd_l resnet50 models for Linf 4/255
python3 ./analysis/plotting/plot_revadv.py \
    --dataset mtsd_l \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 4/255 \
    --baseline-train-eps 8/255 \
    --test-eps 4/255 \
    --baseline-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --trunk-models \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades4_255ft__20210727_0849/resnet50_mtsd_l_trades4_255.pt \
        ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-models \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades4_255ft__20210727_0849/resnet50_mtsd_l_trades4_255.pt \
        ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_stdaug_trades4_255ft__20210728_0005/resnet50_mtsd_l_stdaug_trades4_255.pt \
        ./models/mrevadv/mtsd_l/Linf/4_255/mra4_255__resnet50_mtsd_l_trades8_255 \
        ./models/mrevadv/mtsd_l/Linf/4_255/mra4_255_stdaug__resnet50_mtsd_l_trades8_255 \
    --branch-model-id ResNet50 \
    --only-comp
