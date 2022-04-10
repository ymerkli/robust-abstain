import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import robustabstain.utils.args_factory as args_factory


def get_args():
    parser = args_factory.get_parser(
        description='Plot curves for finetuned and revadv trained models.',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'test-eps', 'adv-norm']
    )
    parser.add_argument(
        '--finetune-model-dir', type=str, required=True, help='Model directory of finetuned model.'
    )
    parser.add_argument(
        '--revadv-model-dir', type=str, required=True, help='Model directory of revadv trained model.'
    )

    print('==> argparsing')
    args = parser.parse_args()

    # post process
    assert len(args.test_eps) == 1, 'Error: specify 1 test-eps'

    return args


def tbevent2df(dpath: str) -> pd.DataFrame:
    """Read tensorboard eventfile and extract data to pd.DataFrame.

    Args:
        dpath (str): Path to model directory.

    Returns:
        pd.DataFrame: Dataframe with event data.
    """
    df = None
    for fname in os.listdir(dpath):
        ea = EventAccumulator(os.path.join(dpath, fname)).Reload()
        tags = ea.Tags()['scalars']
        if len(tags) == 0:
            continue

        out = {}
        for tag in tags:
            tag_values, wall_time, steps = [], [], []
            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)
            out[tag] = pd.DataFrame(data=dict(zip(steps, np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])
        df= pd.concat(out.values(),keys=out.keys())

    return df


def plot(
        args: object, finetune_values: np.ndarray, revadv_values: np.ndarray, beta_values: np.ndarray,
        ylabel: str, title: str, out_filename: str
    ) -> None:
    n_epochs = min(len(finetune_values), len(revadv_values))
    epochs = np.array(range(n_epochs))
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel(ylabel)

    ax1.plot(epochs, finetune_values[:n_epochs], color='blue', label='TRADES finetuned')
    ax1.plot(epochs, revadv_values[:n_epochs], color='orange', label='REVADV finetuned')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(r"$\beta_{revadv}$")
    ax2.set_yscale('log')
    ax2.plot(epochs, beta_values, color='red', label=r"$\beta_{revadv}$")

    #plt.title(title)
    loc = 'upper right' if revadv_values[-1] < revadv_values[0] else 'center right'
    ax1.legend(loc=loc)
    fig.tight_layout()

    # save figure
    plot_dir = os.path.join(args.revadv_model_dir, 'plot')
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, out_filename+'.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, out_filename+'.pdf'))


def main():
    args = get_args()
    finetune_log = tbevent2df(args.finetune_model_dir)
    revadv_log = tbevent2df(args.revadv_model_dir)

    finetune_adv_acc_col, revadv_adv_acc_col = None, None
    for idx_tup in finetune_log.index:
        match = re.match(r'^test/adv_acc\S+$', idx_tup[0])
        if match: # and args.test_eps[0] in idx_tup[0]:
            finetune_adv_acc_col = idx_tup[0]
    for idx_tup in revadv_log.index:
        match = re.match(r'^test/adv_acc\S+$', idx_tup[0])
        if match: # and args.test_eps[0] in idx_tup[0]:
            revadv_adv_acc_col = idx_tup[0]

    assert 'test/rob_inacc' in finetune_log.index, f"{args.finetune_model_dir} eventfile misses 'test/rob_inacc' entry."
    assert 'test/rob_inacc' in revadv_log.index, f"{args.revadv_model_dir} eventfile misses 'test/rob_inacc' entry."
    assert finetune_adv_acc_col, f"{args.finetune_model_dir} eventfile misses 'test/adv_acc' entry."
    assert revadv_adv_acc_col, f"{args.revadv_model_dir} eventfile misses 'test/adv_acc' entry."

    # interpolate to remove NaN
    finetune_log = finetune_log.interpolate(axis=1)
    revadv_log = revadv_log.interpolate(axis=1)

    # plot rob_inacc values
    plot(
        args, finetune_values=finetune_log.loc['test/rob_inacc'].loc['value'].to_numpy(),
        revadv_values=revadv_log.loc['test/rob_inacc'].loc['value'].to_numpy(),
        beta_values=revadv_log.loc['revadv_beta'].loc['value'].to_numpy(),
        ylabel='robust inaccurate %', title='Robust inaccurate fraction', outfile='rob_inacc'
    )

    # plot adv_acc1 values
    plot(
        args, finetune_values=finetune_log.loc[finetune_adv_acc_col].loc['value'].to_numpy(),
        revadv_values=revadv_log.loc[revadv_adv_acc_col].loc['value'].to_numpy(),
        beta_values=revadv_log.loc['revadv_beta'].loc['value'].to_numpy(),
        ylabel='adversarial accuracy %', title='Adversarial accuracy', outfile='adv_acc'
    )


if __name__ == '__main__':
    main()