import csv
import shutil
import pathlib
from os import remove
from os.path import isfile
from collections import OrderedDict

import torch


def load_model(model, ckpt_file, args):
    if args.cuda:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(args.gpuids[0]))
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            model.module.load_state_dict(checkpoint['model'])
    else:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k[:]
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

    return checkpoint


def save_model(state, epoch, is_best, args):
      dir_ckpt = pathlib.Path('checkpoint')
      dir_path = dir_ckpt / args.dataset
      dir_path.mkdir(parents=True, exist_ok=True)

      model_file = dir_path / 'ckpt_epoch_{}.pth'.format(epoch)
      torch.save(state, model_file)

      if is_best:
          shutil.copyfile(model_file, dir_path / 'ckpt_best.pth')

def save_acc(top1,top5,epoch):
    f = open("accuracy.txt",'w')
    top1 = ','.join(map(str,top1))
    top5 = ','.join(map(str,top5))
    epoch = ','.join(map(str,epoch))
    f.write(top1)
    f.write('\n')
    f.write(top5)
    f.write('\n')
    f.write(epoch)

    f.close()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate, decayed rate of 0.04 every 8 epoches"""
    if epoch <=30:
      lr = 0.1
    elif epoch <=60:
      lr = 0.01
    else:
      lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
