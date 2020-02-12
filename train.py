from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import sys
import os
import torch
import argparse
import time
import data
import util
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

import torch.nn.init as init
from models import resnet_bwn, vgg_small, resnet, bireal, bireal_resnet, bireal_act, bireal_act2
from models import bireal32, bireal_32, floatbireal, resnet_naver

from torch.autograd import Variable

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from matplotlib import pyplot as pltz

def shuffle_minibatch(inputs, targets, mixup=True):
    """Shuffle a minibatch and do linear interpolation between images and labels.

    Args:
        inputs: a numpy array of images with size batch_size x H x W x 3.
        targets: a numpy array of labels with size batch_size x 1.
        mixup: a boolen as whether to do mixup or not. If mixup is True, we
            sample the weight from beta distribution using parameter alpha=1,
            beta=1. If mixup is False, we set the weight to be 1 and 0
            respectively for the randomly shuffled mini-batches.
    """
    batch_size = inputs.shape[0]

    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]
    targets1_1 = targets1.unsqueeze(1)

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]
    targets2_1 = targets2.unsqueeze(1)

    y_onehot = torch.FloatTensor(batch_size, 100)
    y_onehot.zero_()
    targets1_oh = y_onehot.scatter_(1, targets1_1, 1)

    y_onehot2 = torch.FloatTensor(batch_size, 100)
    y_onehot2.zero_()
    targets2_oh = y_onehot2.scatter_(1, targets2_1, 1)

    if mixup is True:
        a = numpy.random.beta(1, 1, [batch_size, 1])
    else:
        a = numpy.ones((batch_size, 1))

    b = numpy.tile(a[..., None, None], [1, 3, 32, 32])

    inputs1 = inputs1 * torch.from_numpy(b).float()
    inputs2 = inputs2 * torch.from_numpy(1 - b).float()

    c = numpy.tile(a, [1, 100])

    targets1_oh = targets1_oh.float() * torch.from_numpy(c).float()
    targets2_oh = targets2_oh.float() * torch.from_numpy(1 - c).float()

    inputs_shuffle = inputs1 + inputs2
    targets_shuffle = targets1_oh + targets2_oh

    return inputs_shuffle, targets_shuffle



def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'cifar100/' + str(args.arch) + str(args.trial) + '.pt')


#
# def mixup_data(x, y, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#
#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)
#
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam
#
#
# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

def train_distill(epoch):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if args.distill == 'abound':
        module_list[1].eval()
    elif args.distill == 'factor':
        module_list[2].eval()




    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if args.distill in ['crd']:
            input, target,  contrast_idx = data
        else:
            input, target = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if args.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if args.distill in ['abound']:
            preact = True
        feat_s, logit_s = model(input, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if args.distill == 'kd':
            loss_kd = 0
        elif args.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif args.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif args.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif args.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif args.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif args.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        else:
            raise NotImplementedError(args.distill)

        loss = args.gamma * loss_cls + args.alpha * loss_div + args.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train(epoch):
    model.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        # backwarding
        loss = criterion_cls(output, target)
        loss.backward()


        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))

    return

def mixtrain(epoch):
    model.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs_shuffle, targets_shuffle = shuffle_minibatch(
            data, target, args.mixup)

        inputs_shuffle, targets_shuffle = inputs_shuffle.cuda(), targets_shuffle.cuda()

        optimizer.zero_grad()
        inputs_shuffle, targets_shuffle = Variable(inputs_shuffle), Variable(targets_shuffle)

        output = model(inputs_shuffle)
        # backwarding
        m = nn.LogSoftmax()

        loss = -m(output) * targets_shuffle
        loss = torch.sum(loss) / 128

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))

    return

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion_cls(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * float(correct) / len(test_loader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def adjust_learning_rate(optimizer, epoch):
    lr = 0.25 * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return

if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--cpu', action='store_true', help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',help='dataset path')
    parser.add_argument('--set', default='BWN', type=str)
    parser.add_argument('--arch', action='store', default='nin_xnor',help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,help='the path to the pretrained model')
    parser.add_argument('--teacher', action='store', default=None,help='the path to the teacher model')
    parser.add_argument('--evaluate', action='store_true',help='evaluate the model')
    parser.add_argument('--batch', action='store', default=128,help='insert batch size')
    parser.add_argument('--mixup', action='store_true')


    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    parser.add_argument('--trial', type=str, default='1', help='trial id')
    args = parser.parse_args()
    print('==> Options:', args)

    batch_size = args.batch
    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    if not os.path.isfile(args.data + '/train_data'):
        # check the data path
        raise Exception \
            ('Please assign the correct data path with --data <DATA_PATH>')

    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
    #                          std=(0.2471, 0.2436, 0.2616))
    # ])
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
    #                          std=(0.2471, 0.2436, 0.2616))
    # ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root='../data/cifar100/',
                                     download=True,
                                     transform=transform_train,
                                     train=True)

    test_dataset = datasets.CIFAR100(root='../data/cifar100/',
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)



    # define the model
    print('==> building model', args.arch, '...')

    # choose model nin.vgg,resnet
    if args.arch == 'resnet':
        model = resnet.resnet18()
    elif args.arch == 'resnet34':
        model = resnet.resnet34()
    elif args.arch == 'resnet50':
        model =resnet.resnet50()
    elif args.arch == 'resnet101':
        model = resnet.resnet101()
    elif args.arch == 'resnet152':
        model =resnet.resnet152()
    elif args.arch == 'resnet32':
        model = resnet_bwn.resnet32()
    elif args.arch == 'vgg_small':
        model = vgg_small.Net()
    elif args.arch == 'nin':
        model = nin.Net()
    elif args.arch == 'vgg1x3':
        model = vgg_bwn.Net()
    elif args.arch == 'bireal':
        model = bireal.birealnet18()
        model_t = bireal32.birealnet18()
    elif args.arch == 'birealresnet':
        model = bireal_resnet.birealnet18()
    elif args.arch == 'bireal_act':
        model = bireal_act2.birealnet18()
        model_t = bireal32.birealnet18()
    elif args.arch == 'bireal_act2':
        model = bireal_act2.birealnet18()
        model_t = resnet.resnet18()
    elif args.arch == 'bireal18':
        model = bireal_act2.birealnet18()
    elif args.arch == 'bireal_32':
        model = bireal_32.birealnet18()
    elif args.arch == 'bireal32':
        model = bireal32.birealnet18()
    else:
        raise Exception(args.arch + ' is currently not supported')


    if args.teacher:
        print('==> Load teacher model form', args.teacher, '...')
        teacher_model = torch.load(args.teacher)
        # best_acc = pretrained_model['best_acc']
        best_acc = 0
        print(teacher_model['best_acc'])
        model_t.load_state_dict(teacher_model['state_dict'])

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)

    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        print(pretrained_model['best_acc'])
        model.load_state_dict(pretrained_model['state_dict'])

    print(model)

    module_list = nn.ModuleList([])
    module_list.append(model)


    # define solver and criterion
    base_lr = float(args.lr)

    if args.distill == 'kd':
        criterion_kd = DistillKL(args.kd_T)
    elif args.distill == 'attention':
        criterion_kd = Attention()
    elif args.distill == 'nst':
        criterion_kd = NSTLoss()
    elif args.distill == 'similarity':
        criterion_kd = Similarity()
    elif args.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif args.distill == 'pkt':
        criterion_kd = PKT()
    elif args.distill == 'kdsvd':
        criterion_kd = KDSVD()
    else:
        raise NotImplementedError(args.distill)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    optimizer = optim.SGD(model.parameters(), base_lr, weight_decay=5e-4, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), base_lr, weight_decay=0)
    if args.teacher:
        module_list.append(model_t)
    if not args.cpu:
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True




    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training
    for epoch in range(1, 300):
        adjust_learning_rate(optimizer, epoch)
        if args.teacher :
            train_distill(epoch)
        else :
            print("normal train")
            mixtrain(epoch)
        test(epoch)


#
# # kd
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1
# # FitNet
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill hint --model_s resnet8x4 -a 0 -b 100 --trial 1
# # AT
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill attention --model_s resnet8x4 -a 0 -b 1000 --trial 1
# # SP
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill similarity --model_s resnet8x4 -a 0 -b 3000 --trial 1
# # CC
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill correlation --model_s resnet8x4 -a 0 -b 0.02 --trial 1
# # VID
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill vid --model_s resnet8x4 -a 0 -b 1 --trial 1
# # RKD
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill rkd --model_s resnet8x4 -a 0 -b 1 --trial 1
# # PKT
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill pkt --model_s resnet8x4 -a 0 -b 30000 --trial 1
# # AB
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill abound --model_s resnet8x4 -a 0 -b 1 --trial 1
# # FT
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill factor --model_s resnet8x4 -a 0 -b 200 --trial 1
# # FSP
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill fsp --model_s resnet8x4 -a 0 -b 50 --trial 1
# # NST
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill nst --model_s resnet8x4 -a 0 -b 50 --trial 1
# # CRD
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1
#
# # CRD+KD
# python train_student.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --distill crd --model_s resnet8x4 -a 1 -b 0.8 --trial 1