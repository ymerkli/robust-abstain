TRAIN_EPS='1/255'

python train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --arch cifar_resnet50 \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --arch Carmon2019Unlabeled \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --arch Gowal2020Uncovering_28_10_extra \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

# with data augmentations
python train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --data-aug autoaugment \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
    --arch cifar_resnet50 \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --data-aug autoaugment \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
    --arch Carmon2019Unlabeled \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset cifar10 \
    --data-aug autoaugment \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
    --arch Gowal2020Uncovering_28_10_extra \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune