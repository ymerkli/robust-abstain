python analysis/plotting/plot_revcert.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --noise-sd 0.06 \
    --baseline-noise-sd 0.12 \
    --smoothing-sigma 0.06 \
    --test-eps 0.12 \
    --baseline-model ./models/augm/cifar10/resnet110_cifar10_gaussaugm0.12__20210426_1811/resnet110_cifar10_gaussaugm0.12.pt \
    --trunk-models ./models/std/cifar10/aa_pyramidnet272200_cifar10_std__20210705_0115/aa_pyramidnet272200_cifar10_std.pt \
    --branch-models \
        ./models/augm/cifar10/resnet110_cifar10_gaussaugm0.12__20210426_1811/resnet110_cifar10_gaussaugm0.12.pt \
        ./models/augm/cifar10/resnet110_cifar10_gaussaugm0.06ft__20210808_0158/resnet110_cifar10_gaussaugm0.06.pt \
        ./models/revcertrad/cifar10/0.06/rcr0.06__resnet110_cifar10_gaussaugm0.12 \
    --branch-model-id ResNet110 \
    --revcert-loss revcertrad \
    --plot-varrad \
    --set-title

python analysis/plotting/plot_revcert.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --noise-sd 0.12 \
    --baseline-noise-sd 0.12 \
    --smoothing-sigma 0.12 \
    --test-eps 0.25 \
    --baseline-model ./models/augm/cifar10/resnet110_cifar10_gaussaugm0.12__20210426_1811/resnet110_cifar10_gaussaugm0.12.pt \
    --trunk-models ./models/std/cifar10/aa_pyramidnet272200_cifar10_std__20210705_0115/aa_pyramidnet272200_cifar10_std.pt \
    --branch-models \
        ./models/augm/cifar10/resnet110_cifar10_gaussaugm0.12__20210426_1811/resnet110_cifar10_gaussaugm0.12.pt \
        ./models/revcertrad/cifar10/0.12/rcr0.12__resnet110_cifar10_gaussaugm0.12 \
    --branch-model-id ResNet110 \
    --revcert-loss revcertrad \
    --plot-varrad \
    --set-title

python analysis/plotting/plot_revcert.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --noise-sd 0.06 \
    --baseline-noise-sd 0.50 \
    --smoothing-sigma 0.06 \
    --test-eps 0.12 \
    --baseline-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --trunk-models ./models/std/cifar10/aa_pyramidnet272200_cifar10_std__20210705_0115/aa_pyramidnet272200_cifar10_std.pt \
    --branch-models \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
        ./models/augm/cifar10/Sehwag2021Proxy_R18_cifar10_gaussaugm0.06ft__20210808_0647/Sehwag2021Proxy_R18_cifar10_gaussaugm0.06.pt \
        ./models/revcertrad/cifar10/0.06/rcr0.06__Sehwag2021Proxy_R18 \
    --branch-model-id Sehwag2021 \
    --revcert-loss revcertrad \
    --plot-varrad \
    --set-title

python analysis/plotting/plot_revcert.py \
    --dataset cifar10 \
    --adv-norm L2 \
    --noise-sd 0.12 \
    --baseline-noise-sd 0.50 \
    --smoothing-sigma 0.12 \
    --test-eps 0.25 \
    --baseline-model ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
    --trunk-models ./models/std/cifar10/aa_pyramidnet272200_cifar10_std__20210705_0115/aa_pyramidnet272200_cifar10_std.pt \
    --branch-models \
        ./models/adv/cifar10/L2/Sehwag2021Proxy_R18/Sehwag2021Proxy_R18.pt \
        ./models/augm/cifar10/Sehwag2021Proxy_R18_cifar10_gaussaugm0.12ft__20210808_1045/Sehwag2021Proxy_R18_cifar10_gaussaugm0.12.pt \
        ./models/revcertrad/cifar10/0.12/rcr0.12__Sehwag2021Proxy_R18 \
    --branch-model-id Sehwag2021 \
    --revcert-loss revcertrad \
    --plot-varrad \
    --set-title

