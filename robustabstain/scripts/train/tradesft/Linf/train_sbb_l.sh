TRAIN_EPS='1/255'

python train/train_adv.py \
    --defense trades \
    --dataset sbb_l \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
    --arch resnet50 \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset sbb_l \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/sbb_l/Linf/resnet50_sbb_l_trades8_255__20210625_0038/resnet50_sbb_l_trades8_255.pt \
    --arch resnet50 \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune \
    --data-aug stdaug

