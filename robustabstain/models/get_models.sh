#!/bin/sh
python3 get_robustbench_model.py --dataset cifar10 --adv-norm Linf --arch Carmon2019Unlabeled
python3 get_robustbench_model.py --dataset cifar10 --adv-norm Linf --arch Gowal2020Uncovering_28_10_extra
python3 get_robustbench_model.py --dataset cifar10 --adv-norm L2 --arch Sehwag2021Proxy_R18
python3 get_robustbench_model.py --dataset cifar100 --adv-norm Linf --arch Rebuffi2021Fixing_28_10_cutmix_ddpm