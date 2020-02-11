import argparse

def config():
    parser = argparse.ArgumentParser(description='DenseNet PyTorch')
    parser.add_argument('dataset', metavar='DATA', default='cifar100',
                        help='dataset: cifar10')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (defualt: 0.1)',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=40, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='Path of checkpoint for resuming/testing '
                             'or retraining model (Default: none)')
    parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                        help='Resume model?')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='Test model?')
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='Use cuda?')
    parser.add_argument('-g', '--gpuids', default=[0], nargs='+',
                        help='GPU IDs for using (Default: 0)')
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')

    cfg = parser.parse_args()
    cfg.gpuids = list(map(int, cfg.gpuids))
    return cfg