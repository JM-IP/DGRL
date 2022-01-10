import shutil
import torch
from tensorboardX import SummaryWriter

import sys
import os
import CDbinlosses
import argparse
import data
import util
from data import get_dataset

from preprocess import get_transform
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, './models')
import nin, xnor_resnet, alexnet, binary_connect_network, binary_connect_network_multilayer_V2
from torch.autograd import Variable
parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true',
        help='set if only CPU is available')
parser.add_argument('--data', action='store', default='./data/',
        help='dataset path')
parser.add_argument('--dataset', action='store', default='cifar10',
                    help='dataset path')
parser.add_argument('--arch', action='store', default='resnet',
        help='the architecture for the network: nin')
parser.add_argument('--gpus', default='2',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('--lr', action='store', default='0.0010',
        help='the intial learning rate')
parser.add_argument('--pretrained', action='store', default='./models/vgg_like_real.pth.tar',
        help='the path to the pretrained model')
parser.add_argument('--batch_size', action='store', default='32', type=int,
                    help='batch_size')
parser.add_argument('--workers', action='store', default='8', type=int,
                    help='workers')
parser.add_argument('--alpha_gap', default='0.5', type=float, help='alpha_gap')
parser.add_argument('--alpha_loss_weight', default='0.01', type=float, help='alpha_loss_weight')
parser.add_argument('--shift_gap', default='0.5', type=float, help='shift_gap')
parser.add_argument('--shift_loss_weight', default='0.5', type=float, help='shift_loss_weight')
parser.add_argument('--cor_loss_weight', default='0.001', type=float, help='cor_loss_weight')
parser.add_argument('--M', default='5', type=int, help='multilayer')
parser.add_argument('--N', default='1', type=int, help='multilayer')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='', type=str, help='save_prefix')

args = parser.parse_args()

import logging.config

def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

import time
args = parser.parse_args()
timestr = time.strftime("%Y%m%d-%H%M%S")
args.save = args.save + timestr
# + 'M_is_' + str(args.M) + '__N_is_' + str(args.N) + '__' + "alpha_gap_" + str(args.alpha_gap) + '__' +
os.mkdir(args.save)
setup_logging(os.path.join(args.save, 'log.txt'))
writer = SummaryWriter(os.path.join(args.save, 'graph_logs'))

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '/checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + '/checkpoint.pth.tar', filename + '/model_best.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        # print(-bin_op.target_alpha[0])
        # print(torch.Tensor([0]))
        ################################################################################
        ######### alpha loss ##########
        alpha_loss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        if args.alpha_loss_weight:
            # logging.info('==> using cor loss ...')
            for index in range(0, bin_op.num_of_params, bin_op.M):
                alpha=[]
                for i in range(bin_op.M):
                    weight = bin_op.target_modules[index]
                    n = weight.data[0].nelement()
                    s = weight.data.size()
                    if len(s) == 4:
                        m = weight.norm(1, 3, keepdim=True)\
                                .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
                    elif len(s) == 2:
                        m = weight.norm(1, 1, keepdim=True).div(n)
                    index=index+1
                    alpha.append(m)
                for i in range(1, bin_op.M):
                    alpha_loss = alpha_loss + torch.sum(torch.max(alpha[i] - alpha[i-1]*args.alpha_gap, torch.Tensor([0]).cuda()))
                for i in range(bin_op.M):
                    alpha_loss = alpha_loss + torch.sum(torch.max(-alpha[i], torch.Tensor([0]).cuda()))
        ######### alpha loss end ##########
            # print(alpha_loss)
        # beta_loss = 0
        # for i in range(1, self.N):
        #     beta_loss = beta_loss + max(model.beta[i-1]*0.5 - model.beta[i], 0)
        shift_loss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        if(args.shift_loss_weight):
            # logging.info('==> using shift loss ...')
            for index in range(0, len(bin_op.target_shift), bin_op.N):
                for i in range(1, args.N):
                    shift_loss = shift_loss + torch.max(bin_op.target_shift[index + i - 1] + args.shift_gap - bin_op.target_shift[index + i], torch.Tensor([0]).cuda())


        ################################################################################
        ######### correlation loss ##########
        corloss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        if(args.cor_loss_weight): 
            for i in range(0, bin_op.num_of_params, args.M):
                binweights = [bin_op.target_modules[i].view(-1,1)]
                for j in range(i+1,i+args.M):
            #         print(binweights)
                    binweights.append(bin_op.target_modules[j].view(-1,1))
                binweights = torch.cat(binweights,dim=1)
                # print(a.size())
                corloss = corloss + CDbinlosses.CorrelationPenaltyLoss()(binweights)
        ######### correlation loss end ##########

        accloss = criterion(output, target)
        loss = accloss + args.alpha_loss_weight * alpha_loss + \
                            args.shift_loss_weight * shift_loss + args.cor_loss_weight * corloss
        # loss = Variable(torch.Tensor([0]), requires_grad=True).cuda()
        # print(alpha_loss)
        
        loss.backward()
        # print(bin_op.target_modules[0].grad)
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        # print(model.module.bconv2.conv[0].weight.grad)
        # print(model.module.conv1.weight.grad)
        
        optimizer.step()
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tshiftLoss: {:.6f}\talphaLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), #loss.data.item(),
                accloss.item(), shift_loss.item(), alpha_loss.item(),
                optimizer.param_groups[0]['lr']))


    writer.add_scalar('Train/Loss', accloss.item(), epoch)
    writer.add_scalar('Train/shiftLoss', shift_loss.item(), epoch)

    writer.add_scalar('Train/CorLoss', corloss.item(), epoch)
    writer.add_scalar('Train/alphaLoss', alpha_loss.item(), epoch)
        # writer.add_scalar('Train/Acc', 100. * correct.item() / total, epoch)
    return


def test(arch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        test_loss += criterion(output, target).item()
        # criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    # save_state(model, best_acc, arch, filename = args.save)
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)
    
    test_loss /= len(testloader.dataset)
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    logging.info('Best Accuracy: {:.2f}%\n'.format(best_acc))
    writer.add_scalar('Test/Loss', test_loss, epoch)
    writer.add_scalar('Test/Acc', 100. * float(correct) / len(testloader.dataset), epoch)

    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [70, 140, 210, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


if __name__=='__main__':
    # prepare the options

    args.gpus = [int(i) for i in args.gpus.split(',')]
    torch.cuda.set_device(args.gpus[0])
    logging.info("using gpu ")
    logging.info(torch.cuda.current_device())
    logging.info('==> Options:')
    logging.info(args)
    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    logging.info('==> building model' + args.arch + '...')
    if args.arch == 'nin':
        model = nin.Net()
    elif args.arch == 'resnet':
        model = xnor_resnet.resnet(**{'dataset': 'cifar10', 'num_classes': 10, 'depth': 18})
    elif args.arch == 'alexnet':
        model = alexnet.alexnet()
        default_transform = {
            'train': get_transform('cifar10',
                                   input_size=32, augment=True),
            'eval': get_transform('cifar10',
                                  input_size=32, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)
        regime = getattr(model, 'regime', {0: {'optimizer': 'SGD',
                                               'lr': 0.01,
                                               'momentum': 0.9,
                                               'weight_decay': 0}})
        # define loss function (criterion) and optimizer
        criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    elif args.arch == 'vgg_like':
        model = binary_connect_network.vgg_like()
        default_transform = {
            'train': get_transform('cifar10',
                                   input_size=32, augment=True),
            'eval': get_transform('cifar10',
                                  input_size=32, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)
        regime = getattr(model, 'regime', {0: {'optimizer': 'SGD',
                                               'lr': 0.01,
                                               'momentum': 0.9,
                                               'weight_decay': 0}})
        # define loss function (criterion) and optimizer
        criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    elif args.arch == 'vgg_like_multilayer':
        model = binary_connect_network_multilayer_V2.vgg_like_multilayer(M=args.M, N=args.N, num_classes=10)
        default_transform = {
            'train': get_transform('cifar10',
                                   input_size=32, augment=True),
            'eval': get_transform('cifar10',
                                  input_size=32, augment=False)
        }
        transform = getattr(model, 'input_transform', default_transform)
        regime = getattr(model, 'regime', {0: {'optimizer': 'SGD',
                                               'lr': 0.01,
                                               'momentum': 0.9,
                                               'weight_decay': 0}})
        # define loss function (criterion) and optimizer
        criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    else:
        raise Exception(args.arch+' is currently not supported')

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    testloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # initialize the model
    # if not args.pretrained:
    logging.info('==> Initializing model parameters ...')
    best_acc = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)
            if m.bias is not None:
                m.bias.data.zero_()
    # else:
    #     logging.info('==> Load pretrained model form', args.pretrained, '...')
    #     pretrained_model = torch.load(args.pretrained)
    #     best_acc = pretrained_model['best_acc']
    #     model.load_state_dict(pretrained_model['state_dict'])
    bin_op = util.BinOp_V2(model, args.M, args.N, args.pretrained)

    if not args.cpu:
        logging.info('==> using gpu ...')
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=args.gpus)

    logging.info(model)
    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []
    for key, value in param_dict.items():
        if ('shift' in key):
            params += [{'params': [value], 'lr': base_lr*0,
                        'weight_decay': 0}]
        else:
            params += [{'params':[value], 'lr': base_lr,
                'weight_decay':0.00001}]

    optimizer = optim.Adam(params, lr=base_lr,weight_decay=0.00001)
    args.start_epoch=1
    if args.pretrained:
        bin_op.binarization_pre()
        bin_op.restore()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)
    if(args.cor_loss_weight): 
        logging.info("using cor loss")
    # start training
    for epoch in range(args.start_epoch, 280):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(args.arch)


    writer.close()
    os.system("python2 sentemail.py '" + args.save + "with acc as: " + str(best_acc) + "'")