import os
import setGPU
import re
import warnings
from pathlib import Path

import robustabstain.utils.args_factory as args_factory
from robustabstain.eval.common import PERTURBATION_REGIONS
from robustabstain.utils.regex import ABSTAIN_MODEL_DIR_RE, MODEL_DIR_RE


def get_args():
    parser = args_factory.get_parser(
        description='Evaluate all models in a given directory',
        arg_lists=[
            args_factory.TRAINING_ARGS, args_factory.TESTING_ARGS, args_factory.LOADER_ARGS, args_factory.ATTACK_ARGS,
            args_factory.COMP_ARGS, args_factory.SMOOTHING_ARGS
        ],
    )
    parser.add_argument(
        '--eval-dir', type=str, required=True, help='Parent directory containing model dirs'
    )

    print('==> argparsing')
    args = parser.parse_args()

    # processing
    args = args_factory.process_args(args)

    if args.adv_attack == 'autoattack':
        # autoattack does not take step size or number of steps. Set this args to None to clarify the logs
        args.test_att_n_steps = None
        args.test_att_step_size = None

    # default test epsilons
    if not args.test_eps:
        args.test_eps = PERTURBATION_REGIONS[args.adv_norm]

    return args


def main():
    args = get_args()

    # iterate over models in eval-dir
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        warnings.warn(f'Directory {args.eval_dir} does not exist, terminating...')
        return

    model_dirs = [p for p in eval_dir.iterdir() if p.is_dir()]
    smoothing_sigma = None
    for model_dir in model_dirs:
        abstain_match = re.match(ABSTAIN_MODEL_DIR_RE, model_dir.name)
        model_name = None
        if abstain_match:
            model_name = abstain_match.group('name')
            if abstain_match.group('dataaug'):
                model_name += f"_{abstain_match.group('dataaug')}"
            if abstain_match.group('mode'):
                model_name += f"_{abstain_match.group('mode')}"
        else:
            model_name = model_dir.name

        # finetune models have a small annotation that needs to be removed
        if model_name[-2:] == 'ft':
            model_name = model_name[:-2]

        model_path = model_dir / Path(f'{model_name}.pt')
        if not model_path.is_file():
            # if no model is found under the directory, skip
            raise ValueError(f'Error: no model {model_path} found.')

        # run eval.py script
        cmd = f"python eval/run_solo.py \
            --dataset {args.dataset} \
            --eval-set {args.eval_set} \
            --evals {' '.join(args.evals)} \
            --test-adv-attack {args.adv_attack} \
            --adv-norm {args.adv_norm} \
            --test-eps {' '.join(args.test_eps)} \
            --model {model_path} \
        "

        if args.use_exist_log:
            # use existing log file
            cmd += f"\
                --use-exist-log \
            "

        if args.adv_attack == 'autoattack':
            # autoattack is only run on 1000 samples on train set
            cmd += f"\
                --n-train-samples 1000 \
            "

        if args.smooth:
            assert args.smoothing_sigma, 'Error: specify --smoothing-sigma for smooth eval'
            cmd += f"\
                --smooth \
                --smoothing-sigma {smoothing_sigma} \
            "
        if args.no_eval_report:
            cmd += f"\
                --no-eval-report \
            "

        if any('smo' in s for s in args.evals):
            cmd += f"\
                --n-test-samples 500 \
            "

        # run the evaluation
        os.system(cmd)


if __name__ == '__main__':
    main()
