import json
import argparse
from optims.config import normalize_cflat_args


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json
    normalize_cflat_args(args)

    from trainer import train
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms with C-Flat')
    parser.add_argument('--config', type=str, default='./exps/wa.json', help='Json file of settings')
    cflat_group = parser.add_mutually_exclusive_group()
    cflat_group.add_argument(
        '--cflat-mode', choices=['off', 'cflat', 'cflat-plus'], help='Select the flatness optimizer mode'
    )
    cflat_group.add_argument(
        '--cflat', dest='cflat_mode', action='store_const', const='cflat', help='Enable C-Flat'
    )
    cflat_group.add_argument(
        '--cflat-plus', dest='cflat_mode', action='store_const', const='cflat-plus', help='Enable C-Flat++'
    )
    parser.add_argument('--rho', type=float, default=0.2, help='Perturbation radius')
    parser.add_argument('--lamb', type=float, default=0.2, help='First-order smoothing coefficient')
    parser.add_argument('--cflat-a', '--cflat-A', dest='cflat_A', type=float, default=None,
                        help='C-Flat++ adaptive threshold amplitude')
    parser.add_argument('--cflat-k', dest='cflat_k', type=float, default=None,
                        help='C-Flat++ sigmoid growth rate')
    parser.add_argument('--cflat-t0', dest='cflat_t0', type=int, default=None,
                        help='C-Flat++ sigmoid midpoint step')
    parser.add_argument('--cflat-cof', dest='cflat_cof', type=float, default=None,
                        help='C-Flat++ threshold feedback coefficient')

    return parser


if __name__ == '__main__':
    main()
