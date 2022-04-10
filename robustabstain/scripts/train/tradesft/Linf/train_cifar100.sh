TRAIN_EPS='1/255'

python train/train_adv.py \
    --defense trades \
    --dataset cifar100 \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --arch Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset cifar100 \
    --data-aug autoaugment \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
    --arch Rebuffi2021Fixing_28_10_cutmix_ddpm \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune
