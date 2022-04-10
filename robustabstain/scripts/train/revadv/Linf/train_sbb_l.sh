TRAIN_EPS='1/255'

# train without data augmentations
python train/train_revadv.py \
    --dataset sbb_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
    --dataset sbb_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
    --dataset sbb_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
    --dataset sbb_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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



# train with data augmentations
python train/train_revadv.py \
    --dataset sbb_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
    --dataset sbb_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
    --dataset sbb_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
    --dataset sbb_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/sbb_l/resnet50_sbb_l_wstd__20210810_0037/resnet50_sbb_l_wstd.pt \
    --branch-model ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
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
