TRAIN_EPS='1/255'

# train without data augmentations
python train/train_revadv.py \
    --dataset mtsd_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 5.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset mtsd_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset mtsd_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 0.5 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset mtsd_l \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 0.1 \
    --revadv-beta-gamma 1.0



# train with data augmentations
python train/train_revadv.py \
    --dataset mtsd_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 5.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset mtsd_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 1.0 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset mtsd_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 0.5 \
    --revadv-beta-gamma 1.0

python train/train_revadv.py \
    --dataset mtsd_l \
    --data-aug stdaug \
    --adv-attack pgd \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --test-eps $TRAIN_EPS \
    --trunk-models ./models/std/mtsd_l/resnet50_mtsd_l_std__20210516_0140/resnet50_mtsd_l_std.pt \
    --branch-model ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --epochs 50 \
    --lr 0.005 \
    --lr-sched trades \
    --val-freq 5 \
    --test-freq 5 \
    --train-batch 200 \
    --running-checkpoint \
    --revadv-loss mrevadv \
    --revadv-beta 0.1 \
    --revadv-beta-gamma 1.0
