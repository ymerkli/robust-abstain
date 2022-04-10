python3 analysis/plotting/plot_robacc_heatmap.py \
    --dataset cifar10 \
    --adv-norm Linf \
    --test-eps 1/255 2/255 \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades1_255ft__20210523_1551/resnet50_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades1_255ft__20210525_1434/Carmon2019Unlabeled_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255ft__20210524_1405/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades2_255ft__20210523_1551/resnet50_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades2_255ft__20210529_1410/Carmon2019Unlabeled_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255ft__20210529_1409/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \

python3 analysis/plotting/plot_robacc_heatmap.py \
    --dataset cifar10 \
    --adv-norm Linf \
    --test-eps 4/255 8/255 \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades4_255__20210412_1042/resnet50_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades4_255ft__20210726_0613/Carmon2019Unlabeled_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255ft__20210623_2211/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \

python3 analysis/plotting/plot_robacc_heatmap.py \
    --dataset cifar10 \
    --adv-norm Linf \
    --test-eps 1/255 2/255 4/255 8/255 \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades1_255ft__20210523_1551/resnet50_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades1_255ft__20210525_1434/Carmon2019Unlabeled_cifar10_trades1_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255ft__20210524_1405/Gowal2020Uncovering_28_10_extra_cifar10_trades1_255.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades2_255ft__20210523_1551/resnet50_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades2_255ft__20210529_1410/Carmon2019Unlabeled_cifar10_trades2_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255ft__20210529_1409/Gowal2020Uncovering_28_10_extra_cifar10_trades2_255.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades4_255__20210412_1042/resnet50_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled_cifar10_trades4_255ft__20210726_0613/Carmon2019Unlabeled_cifar10_trades4_255.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255ft__20210623_2211/Gowal2020Uncovering_28_10_extra_cifar10_trades4_255.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \
    --branch-models \
        ./models/adv/cifar10/Linf/resnet50_cifar10_trades8_255__20210410_2245/resnet50_cifar10_trades8_255.pt \
        ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt \
        ./models/adv/cifar10/Linf/Gowal2020Uncovering_28_10_extra/Gowal2020Uncovering_28_10_extra.pt \
        ./models/std/cifar10/resnet50_cifar10_std__20210409_0106/resnet50_cifar10_std.pt \
    --branch-model-ids \
        "ResNet50 [T]" \
        "Carmon2019" \
        "Gowal2020" \
        "ResNet50 [S]" \