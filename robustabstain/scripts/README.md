# scripts

The scripts contained in this directory are streamlined evaluation scripts to
easily reproduce our results. Note that the scripts need to be executed from
the root package directory (i.e. the parent directory of `scripts/`).

```
scripts/
├── eval        # evaluate robustness/accuracy/etc. for trained models
├── plot        # create plots
└── train       # train models via standard/adversarial/noise/abstain training
```


## Usage

### Eval

Evaluation scripts automatically evaluate natural accuracy, robust accuracy, robust coverage, compositional accuracy, etc. of exported models.
Each evaluation run creates a report file `{dataset}_testset_report.json` that lists the respective metrics,
and a log file `testset_{dataset}_{advnorm}_{adv_attack}.csv` that indicates accuracy, robustness, confidence, etc. for every test sample in the dataset.
These log files allow to run plotting scripts without actually requiring the exported models.

Evaluating an exported model from scratch can be done using `robust-abstain/robustabstain/eval/run_*.py` scripts.
As an example, run the following command to evaluate the natural accuracy and adversarial accuracy for $l_{\infty}$ perturbations of radius 1/255, 2/255, 4/255. 8/255, 16/255, using APGD, for the adversarially trained CIFAR-10 model by Carmon et. al.:

```bash
python3 eval/run_solo.py \
    --dataset cifar10 \
    --eval-set test \
    --evals nat adv \
    --test-adv-attack apgd \
    --adv-norm Linf \
    --test-eps 1/255 2/255 4/255 8/255 16/255 \
    --model ./models/adv/cifar10/Linf/Carmon2019Unlabeled/Carmon2019Unlabeled.pt
```

### Plot

The plotting scripts can be used directly since they only require model logs, which we provide in this repository.
As an example, run the following script to plot the results of $l_{\infty}$ trained empirical robustness indicator abstain models
and the comparison to softmax response (SR) and selection network (SN) abstain models, for the CIFAR-10 dataset:

```bash
bash scripts/plot/revadv/cifar10/comp2_cifar10_Linf.sh
```

### Train

We provide training scripts that finetune base models using adversarial training, noise training and our proposed abstain training.
The scripts can be found in `robust-abstain/robustabstain/scripts/train/`.
For example, the script `robust-abstain/robustabstain/scripts/train/revadv/Linf/train_cifar10.sh` can be used
to train all consider CIFAR-10 base models for a specified $l_{\infty}$ perturbation region using our proposed abstain loss $L_{ERA}$. The $\beta$ parameter is set to 1 in the script,
but can be varied to tradeoff robust coverage and robust accuracy.

```bash
bash scripts/train/revadv/Linf/train_cifar10.sh
```
