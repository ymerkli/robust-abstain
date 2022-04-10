import os
from robustbench import load_model

import robustabstain.utils.args_factory as args_factory
from robustabstain.utils.checkpointing import save_checkpoint
from robustabstain.utils.paths import get_root_package_dir


def get_args():
    parser = args_factory.get_parser(
        description='Quickly download and export a model from robustbench.',
        arg_lists=['dataset', 'arch', 'adv-norm'],
        required_args=['dataset', 'arch', 'adv-norm']
    )
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    return args


def main():
    """Download a robustbench model and store it to correct model directory.
    """
    args = get_args()

    base_dir = os.path.join(get_root_package_dir(), 'models', 'adv')
    out_dir = os.path.join(base_dir, args.dataset, args.adv_norm)
    checkpoint_dir = os.path.join(out_dir, args.arch)
    checkpoint_path = os.path.join(checkpoint_dir, f'{args.arch}.pt')

    model = load_model(args.arch, base_dir, args.dataset, args.adv_norm)

    if os.path.exists(os.path.join(out_dir, f'{args.arch}.pt')):
        os.remove(os.path.join(out_dir, f'{args.arch}.pt'))

    save_checkpoint(
        checkpoint_path, model, args.arch, args.dataset, epoch=0,
        optimizer=None, add_state={'adv_norm': args.adv_norm}
    )


if __name__ == '__main__':
    main()