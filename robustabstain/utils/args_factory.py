import argparse
import os
from enum import Enum
from typing import List, Union

from robustabstain.abstain.selector import ABSTAIN_METHODS
from robustabstain.attacker.wrapper import ATTACKS, ADVERSARIAL_NORMS
from robustabstain.loss.revadv import REVADV_LOSSES
from robustabstain.loss.revcert import REVCERT_LOSSES
from robustabstain.utils.helpers import has_attr, loggable_floatstr
from robustabstain.utils.schedulers import LR_SCHEDULERS
from robustabstain.utils.data_utils import DATASETS, DATA_SPLITS
from robustabstain.utils.transforms import DATA_AUG


class ParserAction(Enum):
    STORE_TRUE = 'store_true'
    STORE_FALSE = 'store_false'


"""
Default hyperparameters for training models.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
TRAINING_ARGS = [
    ['arch', str, None, 'Model architecture to use', None],
    ['epochs', int, None, 'Number of epochs to train for', None],
    ['dry-run', ParserAction.STORE_TRUE, None, 'Stop train and test iterations early', None],
    ['resume', str, None, 'Resume training from checkpoint filepath', None],
    ['opt', str, None, 'Optimizer to use', 'sgd'],
    ['load-opt', ParserAction.STORE_TRUE, None, 'If set, load optimizer from checkpoint', None],
    ['resume-opt', ParserAction.STORE_TRUE, None, 'If set and resume is given, optimizer state dict will be loaded from checkpoint', None],
    ['lr', float, None, 'Training learning rate', 0.03],
    ['lr-sched', LR_SCHEDULERS, None, 'Choice of learning rate scheduling', 'trades'],
    ['lr-step', int, None, 'Number of epochs between lr updates', 20],
    ['noise-sd', str, None, 'Gaussian noise sigma for gaussaugm training.', None],
    ['lr-gamma', float, None, 'Factor by which to decrease lr in StepLR', 0.5],
    ['val-freq', int, None, 'How frequently (epochs) to evaluate and log current model on validation set', 5],
    ['test-freq', int, None, 'How frequently (epochs) to evaluate and log current model on test set', -1],
    ['weight-decay', float, None, 'Weight decay', 1e-4],
    ['momentum', float, None, 'Momentum', 0.9],
    ['finetune', ParserAction.STORE_TRUE, None, 'Indicates that an existing model is finetuned.', None],
    ['feature-extractor', ParserAction.STORE_TRUE, None, 'If set, model is used as feature extractor, only last FC layer is trained', None],
    ['running-checkpoint', ParserAction.STORE_TRUE, None, 'If set, model is checkpointed every args.test_freq epoch.', None],
    ['reset-epochs', ParserAction.STORE_TRUE, None, 'If set, epoch counter starts at 0 even when resuming', None],
    ['weighted-loss', ParserAction.STORE_TRUE, None, 'If set, loss is weighted by relative label weights.', None],
    ['seed', int, None, 'Pytorch random seed.', 0]
]


"""
Default hyperparameters for testing/ evaluating models.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
TESTING_ARGS = [
    ['arch', str, None, 'Model architecture to use', None],
    ['dry-run', ParserAction.STORE_TRUE, None, 'Stop train and test iterations early', None],
    ['model', str, None, 'Load model from checkpoint filepath(s)', ''],
    ['eval-att-batches', int, None, 'Number of test batches to run adv attacks on at eval time.', 5],
    ['evals', ['nat', 'adv', 'smo'], '+', 'Types of evaluations to run.', ['nat', 'adv']],
    ['eval-set', DATA_SPLITS, None, 'Dataset split to be evaluated.', 'test'],
    ['no-sample-log', ParserAction.STORE_TRUE, None, 'If set, sample_logs will not be written.', None],
    ['no-eval-report', ParserAction.STORE_TRUE, None, 'If set, no eval report jsons will be written.', None],
    ['use-exist-log', ParserAction.STORE_TRUE, None, 'If set and an existing eval log is found, this log will be used.', None],
    ['seed', int, None, 'Pytorch random seed.', 0]
]


"""
Default hyperparameters for loading data and models.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
LOADER_ARGS = [
    ['data-dir', str, None, 'Data root directory', './data'],
    ['dataset', DATASETS, None, 'Dataset to use', None],
    ['eval-set', DATA_SPLITS, None, 'Dataset split to be evaluated.', 'test'],
    ['test-dataset', DATASETS, None, 'Dataset to use for testset. If not given, --dataset is used.', None],
    ['train-batch', int, None, 'Batch size for training', 100],
    ['test-batch', int, None, 'Batch size for testing', 100],
    ['n-train-samples', int, None, 'Number of train samples', None],
    ['n-test-samples', int, None, 'Number of test samples for an evaluation', None],
    ['num-workers', int, None, 'Number of workers for data loaders', 4],
    ['data-aug', DATA_AUG, None, 'Data augmentation method to use.', None],
    ['no-normalize', ParserAction.STORE_TRUE, None, 'If set, do not normalize data.', None],
    ['seed', int, None, 'Pytorch random seed.', 0],
    ['sample_synth', float, None, 'Probability with which to sample synthetic sample.', 0.0],
    ['interp_synth', float, None, 'Interpolation factor of the synthetic sample.', 0.0]
]


"""
Default hyperparameters for adversarial attacks.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
ATTACK_ARGS = [
    ['adv-attack', ATTACKS, None, 'Type of adversarial attack for adversarial training.', 'pgd'],
    ['test-adv-attack', ATTACKS, None, 'Type of adversarial attack for evaluation.', 'apgd'],
    ['adv-norm', ADVERSARIAL_NORMS, None, 'Norm of the adversarial region.', None],
    ['adv-train', ParserAction.STORE_TRUE, None, 'If set, use adversarial training', None],
    ['defense', ['stdadv', 'trades', 'gaussaugm', 'smoothadv'], None, 'Type of adversarial defense for training.', 'stdadv'],
    ['random-start', bool, None, 'If True, adversarial attack starting input is randomized.', True],
    ['train-eps', str, None, 'Adversarial attack region epsilon to train with (float or fraction).', None],
    ['test-eps', str, '+', 'Adversarial attack region epsilons to test with (float or fraction).', []],
    ['train-att-n-steps', int, None, 'Number of adversarial attack steps during training.', 10],
    ['train-att-step-size', float, None, 'Adversarial attack relative step size during training.', 0.25],
    ['test-att-n-steps', int, None, 'Number of adversarial attack steps during testing', 40],
    ['test-att-step-size', float, None, 'Adversarial attack relative step size during testing.', 0.1],
    ['warmup', int, None, 'Number of epochs over which the maximum allowed perturbation increases linearly from zero to args.epsilon.', 1],
    ['trades-beta', float, None, 'TRADES beta hyperparameter.', 6.0],
    ['autoattack-version', str, None, 'Version of autoattack to use.', 'reduced'],
    ['noise-sd', str, None, 'Gaussian noise sigma for gaussaugm training.', None],
    ['num-noise-vec',  int, None, 'Number of noise vectors to use for finding adversarial examples in Smooth-Adv.', 1],
    ['topk-noise-vec', int, None, 'Number of topk adversarial noise samples to select in rev_noise_loss.', None],
    ['train-multi-noise', ParserAction.STORE_TRUE, None, 'If set, the weights of the network are optimized using \
        all the noise samples. Otherwise, only one of the samples is used.', None],
    ['no-grad-attack', ParserAction.STORE_TRUE, None, 'Choice of whether to use gradients during attack or do the cheap trick.', None],
    ['init-norm-ddn', float, None, '', 1.0],
    ['gamma-ddn', float, None, '', 0.05]
]


"""
Default hyperparameters for compositional architecture.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
COMP_ARGS = [
    ['selector', ABSTAIN_METHODS, None, 'Selection mechanism for compositional architecture.', 'rob'],
    ['conf-threshold', float, None, 'Confidence threshold for confidence abstain selector.', None],
    ['comp-dir', str, None, 'Path to directory of compositional model', None],
    ['branch-model', str, None, 'Path to branch model in the composition.', None],
    ['trunk-models', str, '+', 'Paths to trunk model(s) in the composition. If multiple paths are present, \
        the trunk models are themselves evaluated as compositional architecure. The trunk model [0] acts as \
        the branch model and trunks [1:n-1] act as trunk models.', []],
    ['branch-model-log', str, '+', 'Per sample eval .csv file(s) for the branch model. Requires one log file per test-eps.', []],
    ['trunk-model-log', str, '+', 'Per sample eval .csv file(s) for the trunk model. Requires one log file per test-eps.', []],
    ['revadv-beta', float, None, 'Beta parameter in revadv loss', 1.0],
    ['revadv-beta-gamma', float, None, 'Multiplicative decay factor for revadv-beta', 0.1],
    ['revadv-beta-step', float, None, 'Epoch period for revadv-beta step decay.', 20],
    ['revadv-loss', REVADV_LOSSES, None, 'Version of revadv loss to use.', 'mrevadv'],
    ['revadv-conf', float, None, 'Confidence parameter in revadv gambler loss.', 1.0],
    ['revcert-loss', REVCERT_LOSSES, None, 'Version of revcert loss to use.', 'smoothmrevadv'],
    ['no-train-trunk', ParserAction.STORE_TRUE, None, 'If set, trunk model is NOT trained.', None],
    ['macer-temp', float, None, 'Temperature parameter in the MACER certified radius formulation.', 4.0]
]


"""
Default hyperparameters for smoothing certification.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
SMOOTHING_ARGS = [
    ['smooth', ParserAction.STORE_TRUE, None, 'If set, the smoothed model will be evaluated.', None],
    ['smoothing-sigma', str, None, 'Smoothing noise hyperparameter.', None],
    ['smoothing-N0', int, None, 'Number of Monte Carlo samples to use for selection in Smoothing.', 100],
    ['smoothing-N', int, None, 'Number of Monto Carlor samples to use for estimation in Smoothing.', 100000],
    ['smoothing-alpha', float, None, 'Smoothing failure probability.', 0.001],
    ['smoothing-batch', int, None, 'Batch size for noisy corruption sampling.', 1000]
]


"""
Default hyperparameters for robust self-training (RST).
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
RST_ARGS = [
    ['rst-loss', ['trades', 'noise'], None, 'Loss function to use for RST', 'trades'],
    ['aux-data-filename', str, None, 'Path to pickle file containing unlabeled data and \
        pseudo-labels used for RST', None],
    ['unsup-fraction', float, None, 'Fraction of unlabeled examples in each batch; \
        implicitly sets the weight of unlabeled data in the loss. \
        If set to -1, batches are sampled from a single pool', 0.5],
    ['aux-take-amount', int, None, 'Number of random aux examples to retain. None retains all aux data.', None],
    ['remove-pseudo-labels', ParserAction.STORE_TRUE, None, 'Performs training without pseudo-labels (rVAT)', None],
    ['train-eval-batches', int, None, 'Maximum number for batches in training set eval', 10],
    ['entropy-weight', float, None, 'Weight on entropy loss', 0.0]
]


"""
Default hyperparameters for integration of ACE architecture (Certify and Predict, https://openreview.net/forum?id=USCNapootw).
These hyperparameters are purely used in order to make the ACE code (https://github.com/eth-sri/ACE) work nicely.
Hyperparamter entries are listed as: [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
Note: NARGS=None is default and gives a single argument NOT a list
"""
ACE_ARGS = [
    ['gate-threshold', float, None, 'Threshold for gate network selection. Entropy is negative.', 0.0],
    ['gate-type', ['net', 'entropy'], None, 'Chose whether gating should be based on entropy or a network', 'net'],
    ['ace-trunk-net', str, None, 'ACE trunk net architecture.', 'efficientnet-b0_pre'],
    ['branch-nets', str, '+', 'ACE branch net architectures.', None],
    ['gate-nets', str, '+', 'ACE gate net architectures.', None],
    ['cert-domain', str, None, 'Certification domain.', None],
    ['n-branches', int, None, 'Number of branches.', 1],
    ['cert-net-dim', int, None, 'Size to scale input image for branch and gate nets.', None],
    ['load-branch-model', str, '+', "Model to load on branches. 'True' will load same model as on trunk.", None],
    ['load-gate-model', str, '+', "Model to load on gates. 'True' will load same model as on trunk.", None],
    ['load-trunk-model', str, None, 'Model to load on trunk', None],
    ['gate-feature-extraction', int, None, "How many (linear) layers to retrain at the end of a loaded gate net", None],
    ['n-rand-proj', int, None, 'Number of random projections', 50]
]


ARGS_LISTS = [
    TRAINING_ARGS,
    TESTING_ARGS,
    LOADER_ARGS,
    ATTACK_ARGS,
    COMP_ARGS,
    SMOOTHING_ARGS,
    RST_ARGS,
    ACE_ARGS
]


def add_arg_to_parser(
        parser: argparse.ArgumentParser, arg_name: str, arg_type: type,
        nargs: str, arg_help: str, arg_default: object, required: bool
    ) -> argparse.ArgumentParser:
    """Add a single argument to a parser.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser object.
        arg_name (str): Name of the argument.
        arg_type (type): Type of the argument.
        nargs (str): Number of values to be passed.
        arg_help (str): Helper string.
        arg_default (object): Default arg value.
        required (bool): Whether the arg is requried.

    Returns:
        argparse.ArgumentParser: Parser with added argument.
    """
    has_choices = (type(arg_type) == list)
    has_action = (type(arg_type) == ParserAction)
    kwargs = {
        'required': required,
        'help': f'{arg_help} (default: {arg_default})',
        'default': arg_default
    }
    if has_action:
        kwargs['action'] = arg_type.value
    else:
        kwargs['type'] = type(arg_type[0]) if has_choices else arg_type
        kwargs['nargs'] = nargs
    if has_choices:
        kwargs['choices'] = arg_type

    try:
        parser.add_argument(f'--{arg_name}', **kwargs)
    except argparse.ArgumentError:
        # argument was already added from another arg_list
        pass

    return parser


def add_args_to_parser(parser: argparse.ArgumentParser, arg_list: List, required_args: List[str]) -> argparse.ArgumentParser:
    """Adds arguments from above argument lists to given parser. Sets default values if given.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser object
        arg_list (List): List containing entries of the form [NAME, TYPE/CHOICES, NARGS, HELP, DEFAULT]
        required_args (List[str]): List of args that are required

    Returns:
        argparse.ArgumentParser: parser with added arguments.
    """
    for arg_name, arg_type, nargs, arg_help, arg_default in arg_list:
        parser = add_arg_to_parser(
            parser, arg_name, arg_type, nargs, arg_help,
            arg_default, required=arg_name in required_args
        )

    return parser


def search_add_arg_to_parser(parser: argparse.ArgumentParser, arg: str, required: bool) -> argparse.ArgumentParser:
    """Search the given arg string in all arg lists and add it to given parser. Sets default values if given.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser object
        arg (str): Name of the argument to search and add
        required (bool): Whether the arg is requried.

    Returns:
        argparse.ArgumentParser: parser with added arguments.
    """
    for arg_list in ARGS_LISTS:
        for arg_name, arg_type, nargs, arg_help, arg_default in arg_list:
            if arg_name == arg:
                parser = add_arg_to_parser(
                    parser, arg_name, arg_type, nargs, arg_help,
                    arg_default, required=required
                )
                return parser

    raise ValueError(f'Error: argument {arg} was not found in any arg list.')


def get_parser(description: str, arg_lists: List[Union[List, str]], required_args: List[str] = []) -> argparse.ArgumentParser:
    """Build from given argument lists argparser.

    Args:
        description (str): Description of the program to be parsed
        arg_lists (list): list of arg lists as given above or list of single args.
        required_args (List[str], optional): list of required args. Defaults to [].

    Returns:
        argparse.ArgumentParser: parser
    """
    parser = argparse.ArgumentParser(description=description)
    for args in arg_lists:
        if isinstance(args, list):
            parser = add_args_to_parser(parser, args, required_args)
        elif isinstance(args, str):
            parser = search_add_arg_to_parser(parser, args, required=args in required_args)
        else:
            raise ValueError(f'Error: unsupported type {type(args)} of args to add.')

    return parser


def get_full_parser(description: str = '') -> argparse.ArgumentParser:
    """Get a parser with all arguments added.

    Args:
        description (str, optional): Description of the program to be parsed. Defaults to ''.

    Returns:
        argparse.ArgumentParser: parser
    """
    parser = get_parser(
        description=description,
        arg_lists=[
            TRAINING_ARGS, TESTING_ARGS, LOADER_ARGS,
            ATTACK_ARGS, COMP_ARGS, SMOOTHING_ARGS,
            RST_ARGS, ACE_ARGS
        ],
    )
    return parser


def check_args(args: object, required_args: List[str]) -> None:
    """Check if all required args are present.

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'
        required_args (List[str]): list of required args
    """
    for arg_name in required_args:
        name = arg_name.replace('-', '_')
        if has_attr(args, name):
            continue
        else:
            raise ValueError(f'Error: {arg_name} is a required argument')


def process_args(args: object) -> object:
    """Process parsed args and perform necessary modifications

    Args:
        args (object): Any object subclass exposing 'setattr` and 'getattr'

    Returns:
        object: args object with processed arguments.
    """
    if has_attr(args, 'test_eps'):
        if not args.test_eps:
            args.test_eps = []
        for idx, test_eps in enumerate(args.test_eps):
            args.test_eps[idx] = loggable_floatstr(test_eps)

    if has_attr(args, 'train_eps') and args.train_eps:
        args.train_eps = loggable_floatstr(args.train_eps)

    if has_attr(args, 'smoothing-sigma'):
        args.smoothing_sigma = loggable_floatstr(args.smoothing_sigma)

    if has_attr(args, 'noise-sd'):
        args.noise_sd = loggable_floatstr(args.noise_sd)

    if has_attr(args, 'model') and not args.model:
        args.model = []

    if has_attr(args, 'eval_set'):
        if args.eval_set == 'business_eval':
            assert 'sbb' in args.dataset, "Error: 'business_eval' split is only available for SBB datasets."

    return args