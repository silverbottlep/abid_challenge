import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import data_utils
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ABID Counting')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lrd','--learning-rate-decay-step', default=10, type=int, metavar='N', help='learning rate decay epoch')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')

best_prec = 0
train_losses = []
val_losses = []
def main():
    global args, best_prec, train_losses, val_errs
    args = parser.parse_args()

    # create model
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch]()

    in_features = net.fc.in_features
    new_fc = nn.Linear(in_features,6)
    net.fc = new_fc

    net.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            train_loss_list = checkpoint['train_loss_list']
            val_acc_list = checkpoint['val_acc_list']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    params = net.parameters()
    snapshot_fname = "snapshots/%s.pth.tar" % args.arch
    snapshot_best_fname = "snapshots/%s_best.pth.tar" % args.arch

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        data_utils.ImageFolderCounting(args.data, '../dataset/counting_train.json', transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        data_utils.ImageFolderCounting(args.data, '../dataset/counting_val.json', transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # evaluate on validation set
    if args.evaluate == True:
        validate(val_loader, net, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, net, criterion, optimizer, epoch)

        # evaluate on validation set
        prec = validate(val_loader, net, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        filename = "snapshots/%s.pth.tar" % args.arch
        torch.save({ 
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_prec': best_prec,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }, snapshot_fname)
        if is_best:
            shutil.copyfile(snapshot_fname,snapshot_best_fname)

def train(train_loader, net, criterion, optimizer, epoch):
    cur_lr = adjust_learning_rate(optimizer, epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        score,idx = torch.max(output.data,1)
        correct = (target==idx)
        acc = float(correct.sum())/input.size(0)

        losses.update(loss.data[0], input.size(0))
        train_acc.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec {train_acc.val:.3f} ({train_acc.avg:.3f})'.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               data_time=data_time, loss=losses, train_acc=train_acc))
        train_losses.append(losses.val)

def validate(val_loader, net, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    val_acc = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        score,idx = torch.max(output.data,1)
        correct = (target==idx)
        acc = float(correct.sum())/input.size(0)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec {val_acc.val:.3f} ({val_acc.avg:.3f})'.format(
               i, len(val_loader), batch_time=batch_time, loss=losses,
               val_acc=val_acc))

    print(' * Prec {val_acc.avg:.3f}'.format(val_acc=val_acc))

    val_acc_list.append(val_acc.avg)
    return val_acc.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 1/10 every args.lrd epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lrd))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
