TRAIN_EPS='1/255'

# train Rebuffi2021 model
python train/train_revadv.py \
    --dataset cifar100 \
    --eval-set test \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5 \
    --epochs 50 \
    --lr 0.0005 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

# train Rebuffi2021 model
python train/train_revadv.py \
    --dataset cifar100 \
    --data-aug autoaugment \
    --eval-set test \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --running-checkpoint \
    --train-batch 200 \
    --val-freq 5 \
    --test-freq 5 \
    --epochs 50 \
    --lr 0.0005 \
    --lr-sched trades \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0
