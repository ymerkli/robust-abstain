import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import matplotlib.pyplot as plt
from typing import Tuple

import robustabstain.utils.args_factory as args_factory
from robustabstain.analysis.plotting.utils.decision_region import decision_region_plot
from robustabstain.attacker.wrapper import AttackerWrapper
from robustabstain.utils.checkpointing import load_checkpoint
from robustabstain.utils.helpers import convert_floatstr
from robustabstain.utils.loaders import get_dataloader
from robustabstain.loss.revadv import revadv_loss
from robustabstain.utils.paths import FIGURES_DIR, get_root_package_dir


torch.manual_seed(13)

def get_args():
    parser = args_factory.get_parser(
        description='Plot curves for finetuned and revadv trained models.',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS,
            args_factory.ATTACK_ARGS, args_factory.SMOOTHING_ARGS
        ],
        required_args=['dataset', 'test-eps', 'adv-norm', 'model']
    )

    print('==> argparsing')
    args = parser.parse_args()

    # post process
    assert len(args.test_eps) == 1, 'Error: specify only one test-eps'

    return args


def find_rob_inacc_sample(
        model: nn.Module, attacker: AttackerWrapper, dataloader: torch.utils.data.DataLoader,
        device: str, skip: int = 0
    ) -> Tuple[torch.tensor, torch.tensor, int]:
    """Find a robust inaccurate sample for the given model under the given attacker.

    Args:
        model (nn.Module): Model to evaluate.
        attacker (AttackerWrapper): Attacker.
        dataloader (torch.utils.data.DataLoader): Dataloader with test data.
        device (str): device.
        skip (int, optional): Number of found samples to skip. Defaults to 0.

    Returns:
        Tuple[torch.tensor, torch.tensor]: Image tensor, ground truth label, sample index.
    """
    # put dataset into shuffled dataloader to get random sample order
    shuffled_dataloader = torch.utils.data.DataLoader(
        dataset=dataloader.dataset, batch_size=dataloader.batch_size, shuffle=True, num_workers=dataloader.num_workers
    )
    n_found = 0
    for _, (inputs, targets, sample_indices) in enumerate(shuffled_dataloader):
        for x, y, sample_idx in zip(inputs, targets, sample_indices):
            x, y = x.to(device), y.to(device)
            x, y = x.unsqueeze(0), y.unsqueeze(0) # add batch dimension

            nat_pred = model(x).argmax(1)
            adv_x = attacker.attack(model, x, nat_pred)
            adv_pred = model(adv_x).argmax(1)
            if nat_pred == adv_pred and nat_pred != y:
                if n_found == skip:
                    print('Found [Sample {}]: y={}, nat_pred={}, adv_pred={}'.format(
                        sample_idx.item(), y.item(), nat_pred.item(), adv_pred.item())
                    )
                    return x, y, sample_idx.item()
                n_found += 1

    raise ValueError('Error: no robust inaccurate samples found.')


def train_revadv_single(
        args: object, model: nn.Module, x: torch.tensor, y: torch.tensor, device: str,
        attacker: AttackerWrapper, soft: bool = False
    ) -> bool:
    """Apply revadv training on a single sample.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'.
        model (nn.Module): Model to train.
        x (torch.tensor): Single sample to train on. Must have a batch dimension.
        y (torch.tensor): Target of the single sample.
        device (str): device.
        attacker (AttackerWrapper): Attacker.
        soft (bool, optional): If set, soft inacc multiplier is used. Defaults to False.
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=args.momentum, weight_decay=args.weight_decay)

    sample_is_robust, sample_is_inacc, epoch = True, True, 0
    while sample_is_robust and sample_is_inacc and epoch < 20:
        model.eval()
        nat_out = model(x)
        nat_pred = nat_out.argmax(1)
        nat_probs = F.softmax(nat_out, dim=1)
        x_adv = attacker.attack(model, x, nat_pred)
        adv_out = model(x_adv)
        adv_pred = adv_out.argmax(1)
        p_inacc = 1 - torch.gather(nat_probs, dim=1, index=y.unsqueeze(1)).squeeze()

        # accuracy and robustness indicators
        is_acc_batch = nat_pred.eq(y).int().cpu().numpy()
        is_rob_batch = adv_pred.eq(nat_pred).int().cpu().numpy()

        model.train()
        #revadv_train_loss = revadv_loss(nat_out, adv_out, y, reduction='none')
        abstain_multiplier = torch.from_numpy((1 - is_acc_batch) & is_rob_batch).to(device)
        if soft:
            abstain_multiplier = p_inacc
        loss = (abstain_multiplier * 1 / F.cross_entropy(adv_out, nat_pred, reduction='none')).mean()
        loss = (1 / F.cross_entropy(adv_out, nat_pred, reduction='none')).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        nat_out = model(x)
        nat_pred = nat_out.argmax(1)
        x_adv = attacker.attack(model, x, nat_pred)
        adv_out = model(x_adv)
        adv_pred = adv_out.argmax(1)

        print('[{}]: loss={}, y={}, nat_pred={}, adv_pred={}'.format(
            epoch, loss.item(), y.item(), nat_pred.item(), adv_pred.item())
        )
        sample_is_robust = (nat_pred == adv_pred).item()
        sample_is_inacc = (nat_pred != y).item()
        epoch += 1
    success = (not sample_is_robust) and sample_is_inacc

    return success


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # argparse
    args = get_args()

    # build dataset
    _, _, test_loader, _, _, num_classes = get_dataloader(
        args, args.dataset, normalize=False, indexed=True, val_set_source='test', val_split=0.2
    )

    # load model
    model, _, _ = load_checkpoint(
        args.model, net=None, arch=None, dataset=args.dataset, device=device,
        normalize=not args.no_normalize, optimizer=None, parallel=True
    )

    # build attacker
    test_eps_float = convert_floatstr(args.test_eps[0])
    attacker = AttackerWrapper(
        args.adv_attack, args.adv_norm, test_eps_float, args.test_att_n_steps,
        rel_step_size=args.test_att_step_size, version=args.autoattack_version, device=device
    )

    success = False
    root_dir = get_root_package_dir()
    while not success:
        x, y, _ = find_rob_inacc_sample(model, attacker, test_loader, device)

        # plot before
        decision_region_plot(
            model, x, y, device, eps=test_eps_float, eps_plot=4 * test_eps_float,
            adv_attack=args.adv_attack, adv_norm=args.adv_norm, att_n_steps=args.test_att_n_steps,
            rel_step_size=args.test_att_step_size, nx=500, ny=500, plotlib='matplotlib',
            savepath=os.path.join(root_dir, FIGURES_DIR, 'illustration/revadv_decision_region_before')
        )

        # make sample x non-robust
        success = train_revadv_single(args, model, x, y, device, attacker, soft=True)

    # plot after
    decision_region_plot(
        model, x, y, device, eps=test_eps_float, eps_plot=4 * test_eps_float,
        adv_attack=args.adv_attack, adv_norm=args.adv_norm, att_n_steps=args.test_att_n_steps,
        rel_step_size=args.test_att_step_size, nx=500, ny=500, plotlib='matplotlib',
        savepath=os.path.join(root_dir, FIGURES_DIR, 'illustration/revadv_decision_region_after')
    )


if __name__ == '__main__':
    main()