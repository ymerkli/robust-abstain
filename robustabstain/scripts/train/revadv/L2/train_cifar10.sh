TRAIN_EPS='0.25'

python train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 5.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 0.5 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 0.1 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 5.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 0.5 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --adv-norm L2 \
    --train-eps $TRAIN_EPS \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --revadv-loss mrevadv \
    --running-checkpoint \
    --revadv-beta 0.1 \
    --revadv-beta-gamma 1.0
