import time
import pathlib
from os.path import isfile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model import DenseNet
from utils import *
from config import config
from data import DataLoader
from matplotlib import pyplot as plt

best_acc1 = 0


def main():
    global args, start_epoch, best_acc1
    args = config()

    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    print('\n=> Build DenseNet..')
    #growthRate,depth,reduction,nClasses,bottleneck)
    model = DenseNet(growthRate = 10)
    print(model)
    print('==> Complete build')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    start_epoch = 0
    n_retrain = 0

    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True

    # checkpoint file
    ckpt_dir = pathlib.Path('checkpoint')
    ckpt_file = ckpt_dir / args.dataset / args.ckpt

    # for resuming training
    if args.resume:
        if isfile(ckpt_file):
            print('\n==> Loading Checkpoint \'{}\''.format(args.ckpt))
            checkpoint = load_model(model, ckpt_file, args)

            start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                args.ckpt, start_epoch))
        else:
            print('==> no checkpoint found \'{}\''.format(
                args.ckpt))
            return

    # Data loading
    print('\n==> Load data..')
    train_loader, val_loader = DataLoader(args.batch_size, args.workers,
                                          args.dataset, args.datapath,
                                          args.cuda)

    # for evaluation
    if args.evaluate:
        if isfile(ckpt_file):
            print('\n==> Loading Checkpoint \'{}\''.format(args.ckpt))
            checkpoint = load_model(model, ckpt_file, args)

            print('==> Loaded Checkpoint \'{}\' (epoch {})'.format(
                args.ckpt, start_epoch))

            # evaluate on validation set
            print('\n===> [ Evaluation ]')
            start_time = time.time()
            acc1, acc5 = validate(val_loader, model, criterion)
            elapsed_time = time.time() - start_time
            print('====> {:.2f} seconds to evaluate this model\n'.format(
                elapsed_time))
            return
        else:
            print('==> no checkpoint found \'{}\''.format(
                args.ckpt))
            return

    # train...
    train_time = 0.0
    validate_time = 0.0
    lr = args.lr
    list_Acc1 = []
    list_Acc5 = []
    list_epoch = []
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr)
        print('\n==> Epoch: {}, lr = {}'.format(
            epoch, optimizer.param_groups[0]["lr"]))

        # train for one epoch
        print('===> [ Training ]')
        start_time = time.time()
        acc1_train, acc5_train = train(train_loader,
            epoch=epoch, model=model,
            criterion=criterion, optimizer=optimizer)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print('====> {:.2f} seconds to train this epoch\n'.format(
            elapsed_time))

        # evaluate on validation set
        print('===> [ Validation ]')
        start_time = time.time()
        acc1_valid, acc5_valid = validate(val_loader, model, criterion)
        elapsed_time = time.time() - start_time
        validate_time += elapsed_time
        print('====> {:.2f} seconds to validate this epoch\n'.format(
            elapsed_time))

        # remember best Acc@1 and save checkpoint
        is_best = acc1_valid > best_acc1
        best_acc1 = max(acc1_valid, best_acc1)
        state = {'epoch': epoch + 1,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        save_model(state, epoch, is_best, args)
        list_Acc1.append(acc1_valid)
        list_Acc5.append(acc5_valid)
        list_epoch.append(epoch)
    plt.plot(list_epoch,list_Acc1)
    plt.plot(list_epoch, list_Acc5)
    print(list_Acc1,list_Acc5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.lengend(['ACC1','ACC5'])
    plt.show()
    avg_train_time = train_time / (args.epochs-start_epoch)
    avg_valid_time = validate_time / (args.epochs-start_epoch)
    total_train_time = train_time + validate_time
    print('====> average training time per epoch: {:,}m {:.2f}s'.format(
        int(avg_train_time//60), avg_train_time%60))
    print('====> average validation time per epoch: {:,}m {:.2f}s'.format(
        int(avg_valid_time//60), avg_valid_time%60))
    print('====> training time: {}h {}m {:.2f}s'.format(
        int(train_time//3600), int((train_time%3600)//60), train_time%60))
    print('====> validation time: {}h {}m {:.2f}s'.format(
        int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
    print('====> total training time: {}h {}m {:.2f}s'.format(
        int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))


def train(train_loader, **kwargs):
    epoch = kwargs.get('epoch')
    model = kwargs.get('model')
    criterion = kwargs.get('criterion')
    optimizer = kwargs.get('optimizer')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

        end = time.time()

    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))
