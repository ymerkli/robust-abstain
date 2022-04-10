TRAIN_EPS='1/255'

# train ResNet50 model
python3 train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5  \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

# train Carmon2019 model
python3 train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model .//models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5  \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

# train Gowal2020 model
python3 train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5  \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

# train ResNet50 model
python3 train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5  \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

# train Carmon2019 model
python3 train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model .//models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5  \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

# train Gowal2020 model
python3 train/train_revadv.py \
    --dataset cifar10 \
    --data-aug autoaugment \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5  \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0
