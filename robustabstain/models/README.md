# Models

Collection of checkpointed models, corresponding evaluation logs.

Note: Generally, models are named according to the convention `{model_architecture}_{dataset}_{defense}_{aug}?.pt` and are exported in the following state dict format:

```
{
    'arch': ...,        # Name of model architecture
    'dataset': ...,     # Name of dataset model was trained for
    'epoch': ...,       # Training epoch at which model was exported
    'model': ...,       # Model state dict
    'optimizer': ...    # Optimizer state dict
}
```

Additional keys may be present depending on the type of model (e.g. keys `adv_norm, adv_prec1` for adversarially trained models).

Models named `checkpoint.pt.+` are generally checkpoints from some training procedure. In case training wants to be resumed for some model using the appropriate training scripts, these model checkpoints should be used and not the above mentioned checkpoints.

## Download models

Several of our base models are taken from [RobustBench](https://github.com/RobustBench/robustbench).
We provide an easy to use bash script that automatically downloads all required models from RobustBench and exports them the correct format.

```bash
bash get_models.sh
```

## Logs

Each evaluation run creates a report file `{dataset}_testset_report.json` that lists the respective metrics,
and a log file `testset_{dataset}_{advnorm}_{adv_attack}.csv` that indicates accuracy, robustness, confidence, etc. for every test sample in the dataset.
Each log file is located in the respective model directory and allows to run plotting scripts without actually requiring the exported models.
