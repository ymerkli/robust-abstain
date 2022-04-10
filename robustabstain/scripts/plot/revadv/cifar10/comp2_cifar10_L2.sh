#!/bin/bash

EVAL_SET=$1
if [ -z "$EVAL_SET" ]
then
    EVAL_SET="test"
fi


python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm L2 \
    --train-eps 0.12 \
    --baseline-train-eps 0.50 \
    --test-eps 0.12 \
    --baseline-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-models \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18_cifar10_trades0.12ft__20210807_1923/Sehwag2021Proxy_R18_cifar10_trades0.12.pt \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18_cifar10_autoaugment_trades0.12ft__20210808_0128/Sehwag2021Proxy_R18_cifar10_autoaugment_trades0.12.pt \
        ./models/mrevadv/cifar10/L2/0.12/mra0.12__Sehwag2021Proxy_R18 \
        ./models/mrevadv/cifar10/L2/0.12/mra0.12_autoaugment__Sehwag2021Proxy_R18 \
    --branch-model-id Sehwag2021 \
    --conf-baseline

python3 analysis/plotting/plot_revadv.py \
    --dataset cifar10 \
    --eval-set $EVAL_SET \
    --adv-norm L2 \
    --train-eps 0.25 \
    --baseline-train-eps 0.50 \
    --test-eps 0.25 \
    --baseline-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-models \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18_cifar10_trades0.25ft__20210715_1424/Sehwag2021Proxy_R18_cifar10_trades0.25.pt \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18_cifar10_autoaugment_trades0.25ft__20210808_0930/Sehwag2021Proxy_R18_cifar10_autoaugment_trades0.25.pt \
        ./models/mrevadv/cifar10/L2/0.25/mra0.25__Sehwag2021Proxy_R18 \
        ./models/mrevadv/cifar10/L2/0.25/mra0.25_autoaugment__Sehwag2021Proxy_R18 \
    --branch-model-id Sehwag2021 \
    --conf-baseline