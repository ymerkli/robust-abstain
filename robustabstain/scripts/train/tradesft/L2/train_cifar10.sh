RESUME_PATH='./models/adv/cifar10/L2/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt'
ARCH='Rebuffi2021Fixing_28_10_cutmix_ddpm'

python3 train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --adv-norm L2 \
    --train-eps 0.12 \
    --test-eps 0.12 \
    --resume $RESUME_PATH \
    --arch $ARCH \
    --epochs 50 \
    --train-batch 200 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python3 train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --data-aug autoaugment \
    --adv-norm L2 \
    --train-eps 0.12 \
    --test-eps 0.12 \
    --resume $RESUME_PATH \
    --arch $ARCH \
    --epochs 50 \
    --train-batch 200 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python3 train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --adv-norm L2 \
    --train-eps 0.25 \
    --test-eps 0.25 \
    --resume $RESUME_PATH \
    --arch $ARCH \
    --epochs 50 \
    --train-batch 200 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python3 train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --data-aug autoaugment \
    --adv-norm L2 \
    --train-eps 0.25 \
    --test-eps 0.25 \
    --resume $RESUME_PATH \
    --arch $ARCH \
    --epochs 50 \
    --train-batch 200 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune