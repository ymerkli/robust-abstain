TRAIN_EPS='1/255'

python train/train_adv.py \
    --defense trades \
    --dataset mtsd_l \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --arch resnet50 \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune

python train/train_adv.py \
    --defense trades \
    --dataset mtsd_l \
    --data-aug stdaug \
    --adv-norm Linf \
    --train-eps $TRAIN_EPS \
    --resume ./models/adv/mtsd_l/Linf/resnet50_mtsd_l_trades8_255__20210518_1118/resnet50_mtsd_l_trades8_255.pt \
    --arch resnet50 \
    --epochs 50 \
    --lr 0.001 \
    --lr-sched trades \
    --finetune
