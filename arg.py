import os
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--root', type=str, required=True,
                        help="root path to data directory")
    parser.add_argument('-d', '--dataset', type=str, required=True, nargs='+',
                        help="deepfashion or fld")

    parser.add_argument('--base_epoch', type=int, default=0,
                        help="base epoch of models to save")
    parser.add_argument('-b', '--batchsize', type=int, default=50,
                        help='batchsize')
    parser.add_argument('--epoch', type=int, default=30,
                        help='the number of epoch')
    parser.add_argument('--decay-epoch', type=int, default=5,
                        help='decay epoch')
    parser.add_argument('-g', '--gamma', type=float, default=0.1,
                        help='decay gamma')
    parser.add_argument('-lr','--learning-rate', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--evaluate', type=int, default=0,
                        help='evaluation only')
    parser.add_argument('-w', '--weight-file', type=str, default=None,
                        help='weight file')
    parser.add_argument('--loss-type', type=str, default='mse',
                        help='loss function type (mse or cross_entropy)')
    parser.add_argument('--cca', type=int, default=1,
                        help='criss-cross attention module')
    parser.add_argument('--update-weight', type=int, default=0)

    return parser

