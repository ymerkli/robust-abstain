python3 train/train_gaussaugm.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --test-eps 0.12 \
    --noise-sd 0.06 \
    --arch cifar_resnet110 \
    --epochs 50 \
    --lr 0.01 \
    --lr-sched trades \
    --finetune \
    --resume \
        ./models/augm/cifar10/resnet110_cifar10_gaussaugm0.12__20210426_1811/resnet110_cifar10_gaussaugm0.12.pt

python3 train/train_gaussaugm.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --test-eps 0.12 \
    --noise-sd 0.06 \
    --arch Sehwag2021Proxy_R18 \
    --epochs 50 \
    --lr 0.01 \
    --lr-sched trades \
    --finetune \
    --resume \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt

python3 train/train_gaussaugm.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --test-eps 0.25 \
    --noise-sd 0.12 \
    --arch Sehwag2021Proxy_R18 \
    --epochs 50 \
    --lr 0.01 \
    --lr-sched trades \
    --finetune \
    --resume \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt
