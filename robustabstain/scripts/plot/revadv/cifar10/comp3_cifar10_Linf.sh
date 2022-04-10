#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi


# Plotting cifar10 resnet50 models for Linf 1/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 1/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 2/255 \
    --test-eps 1/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades1_255ft__20210523_1551/resnet50_cifar10_trades1_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades1_255ft__20210523_1551/resnet50_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/resnet50_cifar10_autoaugment_trades1_255ft__20210620_1824/resnet50_cifar10_autoaugment_trades1_255.pt \
        ./models/mrevadv/cifar10/Linf/1_255/mra1_255__resnet50_cifar10_trades1_255 \
        ./models/mrevadv/cifar10/Linf/1_255/mra1_255_autoaugment__resnet50_cifar10_trades1_255 \
        ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-model-id ResNet50 \
    --ace-model-id Conv3 \
    --only-comp

# Plotting cifar10 Carmon2019Unlabeled models for Linf 1/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 1/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 2/255 \
    --test-eps 1/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades1_255ft__20210525_1434/Carmon2019Unlabeled_cifar10_trades1_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades1_255ft__20210525_1434/Carmon2019Unlabeled_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_autoaugment_trades1_255ft__20210616_1517/Carmon2019Unlabeled_cifar10_autoaugment_trades1_255.pt \
        ./models/mrevadv/cifar10/Linf/1_255/mra1_255__Carmon2019Unlabeled \
        ./models/mrevadv/cifar10/Linf/1_255/mra1_255_autoaugment__Carmon2019Unlabeled \
        ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-model-id Carmon2019 \
    --ace-model-id Conv3 \
    --only-comp

# Plotting cifar10 Gowal2020Uncovering_28_10_extra models for Linf 1/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 1/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 2/255 \
    --test-eps 1/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255ft__20210524_1405/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255ft__20210524_1405/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_autoaugment_trades1_255ft__20210615_1154/Gowal2020Uncovering_28_10_extra_cifar10_autoaugment_trades1_255.pt \
        ./models/mrevadv/cifar10/Linf/1_255/mra1_255__Gowal2020Uncovering_28_10_extra \
        ./models/mrevadv/cifar10/Linf/1_255/mra1_255_autoaugment__Gowal2020Uncovering_28_10_extra \
        ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-model-id Gowal2020 \
    --ace-model-id Conv3 \
    --only-comp



# Plotting cifar10 resnet50 models for Linf 2/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 2/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 2/255 \
    --test-eps 2/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades2_255ft__20210523_1551/resnet50_cifar10_trades2_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades2_255ft__20210523_1551/resnet50_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/resnet50_cifar10_autoaugment_trades2_255ft__20210621_0006/resnet50_cifar10_autoaugment_trades2_255.pt \
        ./models/mrevadv/cifar10/Linf/2_255/mra2_255__resnet50_cifar10_trades2_255 \
        ./models/mrevadv/cifar10/Linf/2_255/mra2_255_autoaugment__resnet50_cifar10_trades2_255 \
        ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-model-id ResNet50 \
    --ace-model-id Conv3 \
    --only-comp

# Plotting cifar10 Carmon2019Unlabeled models for Linf 2/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 2/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 2/255 \
    --test-eps 2/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades2_255ft__20210529_1410/Carmon2019Unlabeled_cifar10_trades2_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades2_255ft__20210529_1410/Carmon2019Unlabeled_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_autoaugment_trades2_255ft__20210615_1203/Carmon2019Unlabeled_cifar10_autoaugment_trades2_255.pt \
        ./models/mrevadv/cifar10/Linf/2_255/mra2_255__Carmon2019Unlabeled \
        ./models/mrevadv/cifar10/Linf/2_255/mra2_255_autoaugment__Carmon2019Unlabeled \
        ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-model-id Carmon2019 \
    --ace-model-id Conv3 \
    --only-comp

# Plotting cifar10 Gowal2020Uncovering_28_10_extra models for Linf 2/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 2/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 2/255 \
    --test-eps 2/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255ft__20210529_1409/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255ft__20210529_1409/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_autoaugment_trades2_255ft__20210616_1516/Gowal2020Uncovering_28_10_extra_cifar10_autoaugment_trades2_255.pt \
        ./models/mrevadv/cifar10/Linf/2_255/mra2_255__Gowal2020Uncovering_28_10_extra \
        ./models/mrevadv/cifar10/Linf/2_255/mra2_255_autoaugment__Gowal2020Uncovering_28_10_extra \
        ./models/ace/cifar10/Linf/ACE_Net_COLT_cert_2_255/C3_ACE_Net_COLT_cert_cifar10_2_255.pt \
    --branch-model-id Gowal2020 \
    --ace-model-id Conv3 \
    --only-comp



# Plotting cifar10 resnet50 models for Linf 4/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 4/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 4/255 \
    --test-eps 4/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades4_255ft__20210725_2249/cifar_resnet50_cifar10_trades4_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades4_255ft__20210725_2249/cifar_resnet50_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/resnet50_cifar10_autoaugment_trades4_255ft__20210726_0328/cifar_resnet50_cifar10_autoaugment_trades4_255.pt \
        ./models/mrevadv/cifar10/Linf/4_255/mra4_255__resnet50_cifar10_trades4_255 \
        ./models/mrevadv/cifar10/Linf/4_255/mra4_255_autoaugment__resnet50_cifar10_trades4_255 \
        ./models/ace/cifar10/Linf/ACE_Net_IBP_cert_8_255/C3_ACE_Net_IBP_cert_cifar10_8_255.pt \
    --branch-model-id ResNet50 \
    --ace-model-id Conv3 \
    --only-comp

# Plotting cifar10 Carmon2019Unlabeled models for Linf 4/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 4/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 4/255 \
    --test-eps 4/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades4_255ft__20210726_0613/Carmon2019Unlabeled_cifar10_trades4_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades4_255ft__20210726_0613/Carmon2019Unlabeled_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_autoaugment_trades4_255ft__20210726_2135/Carmon2019Unlabeled_cifar10_autoaugment_trades4_255.pt \
        ./models/mrevadv/cifar10/Linf/4_255/mra4_255__Carmon2019Unlabeled \
        ./models/mrevadv/cifar10/Linf/4_255/mra4_255_autoaugment__Carmon2019Unlabeled \
        ./models/ace/cifar10/Linf/ACE_Net_IBP_cert_8_255/C3_ACE_Net_IBP_cert_cifar10_8_255.pt \
    --branch-model-id Carmon2019 \
    --ace-model-id Conv3 \
    --only-comp

# Plotting cifar10 Gowal2020Uncovering_28_10_extra models for Linf 4/255"
python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm Linf \
    --train-eps 4/255 \
    --baseline-train-eps 8/255 \
    --ace-train-eps 4/255 \
    --test-eps 4/255 \
    --branch-nets C3_cifar10 \
    --gate-nets C3_cifar10 \
    --n-branches 1 \
    --gate-type net \
    --gate-threshold -0.0 \
    --baseline-model ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --trunk-models \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255ft__20210623_2211/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255.pt \
        ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \ 
    --branch-models \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255ft__20210623_2211/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_autoaugment_trades4_255ft__20210623_0056/Gowal2020Uncovering_28_10_extra_cifar10_autoaugment_trades4_255.pt \
        ./models/mrevadv/cifar10/Linf/4_255/mra4_255__Gowal2020Uncovering_28_10_extra \
        ./models/mrevadv/cifar10/Linf/4_255/mra4_255_autoaugment__Gowal2020Uncovering_28_10_extra \
        ./models/ace/cifar10/Linf/ACE_Net_IBP_cert_8_255/C3_ACE_Net_IBP_cert_cifar10_8_255.pt \
    --branch-model-id Gowal2020 \
    --ace-model-id Conv3 \
    --only-comp
