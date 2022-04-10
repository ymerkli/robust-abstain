# train

This subpackage contains training code that trains models using:

* standard training (`robust-abstain/robustabstain/train/train_std.py`)
* adversarial training (`robust-abstain/robustabstain/train/train_adv.py`)
* Gaussian noise augmentation training (`robust-abstain/robustabstain/train/train_gaussaugm.py`)
* Our proposed abstain training for empirical robustness (`robust-abstain/robustabstain/train/train_revadv.py`)
* Our proposed abstain training for certified robustness (`robust-abstain/robustabstain/train/train_revcert.py`)


As an example, training the base model by Carmon et. al. (taken from [RobustBench](https://github.com/RobustBench/robustbench)) using our proposed empirical robustness abstain loss $L_{ERA}$ on CIFAR-10 for $\epsilon_{\infty} = 2/255$ perturbations is done as follows:

```bash
cd robustabstain
python3 train/train_revadv.py \
    --dataset cifar10 \
    --eval-set test \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps 2/255 \
    --test-eps 2/255 \
    --trunk-models ./models/std/cifar10/wrn4010_cifar10_std__20210919_2248/wrn4010_cifar10_std.pt \
    --branch-model ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
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
```