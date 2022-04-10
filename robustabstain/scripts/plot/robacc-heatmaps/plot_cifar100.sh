python3 analysis/plotting/plot_robacc_heatmap.py \
    --dataset cifar100 \
    --adv-norm Linf \
    --test-eps 1/255 2/255 \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades1_255ft__20210609_2128/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades1_255.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades2_255ft__20210610_2149/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades2_255.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \

python3 analysis/plotting/plot_robacc_heatmap.py \
    --dataset cifar100 \
    --adv-norm Linf \
    --test-eps 4/255 8/255 \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades4_255ft__20210705_2313/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades4_255.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \

python3 analysis/plotting/plot_robacc_heatmap.py \
    --dataset cifar100 \
    --adv-norm Linf \
    --test-eps 1/255 2/255 4/255 8/255 \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades1_255ft__20210609_2128/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades1_255.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades2_255ft__20210610_2149/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades2_255.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades4_255ft__20210705_2313/Rebuffi2021Fixing_28_10_cutmix_ddpm_cifar100_trades4_255.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \
    --branch-models \
        ./models/adv/cifar100/Linf/Rebuffi2021Fixing_28_10_cutmix_ddpm/Rebuffi2021Fixing_28_10_cutmix_ddpm.pt \
        ./models/std/cifar100/wrn2810_cifar100_std__20210503_1059/wrn2810_cifar100_std.pt \
    --branch-model-ids \
        "Rebuffi2021" \
        "WRN-28-10 [S]" \