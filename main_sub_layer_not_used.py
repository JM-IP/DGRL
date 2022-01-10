import torch
import sys
import os
import argparse
import util
from data import get_dataset
from preprocess import get_transform
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, './models')
import nin, xnor_resnet, alexnet
from torch.autograd import Variable

def save_state(model, best_acc, arch):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    # for key in state['state_dict'].keys():
    #     if 'module' in key:
    #         state['state_dict'][key.replace('module.', '')] = \
    #                 state['state_dict'].pop(key)
    torch.save(state, 'models/' + arch + 'sublayer.pth.tar')

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
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), #loss.data.item(),
                loss.data[0],
                optimizer.param_groups[0]['lr']))
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
        test_loss += criterion(output, target).data[0]
        # criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc, arch)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def adjust_learning_rate(optimizer, epoch):
    update_list = [80, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.3
    return


if __name__=='__main__':
    # prepare the options
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
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--batch_size', action='store', default='32', type=int,
                        help='batch_size')
    parser.add_argument('--workers', action='store', default='8', type=int,
                        help='workers')

    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    args.gpus = [int(i) for i in args.gpus.split(',')]
    torch.cuda.set_device(args.gpus[0])
    print("using gpu ", torch.cuda.current_device())
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    # trainset = data.dataset(root=args.data, train=True)
    #
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #         shuffle=True, num_workers=8)
    #
    # testset = data.dataset(root=args.data, train=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #         shuffle=False, num_workers=8)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model', args.arch, '...')
    if args.arch == 'nin':
        model = nin.Net()
    elif args.arch == 'resnet':
        model = xnor_resnet.resnet(**{'dataset': 'cifar10', 'num_classes': 10, 'depth': 18})
    elif args.arch == 'alexnet':
        model = alexnet.alexnet_sublayer()
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
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
        pretrained = './models/alexnet.pth.tar'
        pretrained_model = torch.load(pretrained)
        model_old = alexnet.alexnet()
        model_old.cuda()
        model_old = torch.nn.DataParallel(model_old, device_ids=args.gpus)
        model_old.load_state_dict(pretrained_model['state_dict'])
        best_acc_old = pretrained_model['best_acc']
        print(best_acc_old)
        count = 0

        new_model_state = model.state_dict()
        new_model_list = list(new_model_state.items())
        for name, params in list(model_old.state_dict().items()):
            # print(name)
            #     if isinstance(params, nn.Parameter):
            #         params = params.data

            new_model_state[new_model_list[count][0]] = model_old.state_dict().pop(name)
            # print(new_model_list[count][0])
            # print(new_model_state[new_model_list[count][0]])
            count = count + 1
        model.load_state_dict(new_model_state)
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=args.gpus)

    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

    # optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training
    for epoch in range(1, 250):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test(args.arch)
