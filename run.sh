#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# @Time    : 21/03/19 15:25
# @Author  : Jianming Ip
# @Site    : ${SITE}
# @File    : ${NAME}.py
# @Company : VMC Lab in Peking University

py27 && python main_V2.py --gpus=1 --N=5 --M=5 --save=CIFAR10_M5N5.V6_cor_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0 --cor_loss_weight=0.001
py3 && python main_real.py --gpus=0 --N=5 --M=5 --save=CIFAR10_M5N5.V7_binact_pretrianed --pretrained=/home/yjm/XNOR-Net-Pytorch/CIFAR_10/models/vgg_like_real.pth.tar  --arch=vgg_like_real_binact
py3 && python main_real.py --gpus=1 --N=3 --M=3 --save=CIFAR10_M3N3.V7_binact_pretrianed --pretrained=/home/yjm/XNOR-Net-Pytorch/CIFAR_10/models/vgg_like_real.pth.tar  --arch=vgg_like_real_binact
py3 && python main_real.py --gpus=2 --N=2 --M=2 --save=CIFAR10_M2N2.V7_binact_pretrianed --pretrained=/home/yjm/XNOR-Net-Pytorch/CIFAR_10/models/vgg_like_real.pth.tar --arch=vgg_like_real_binact
py3 && python main_real.py --gpus=3 --N=5 --M=5 --save=CIFAR10_binact_N5M5.V8 --arch=vgg_like_real_binact

py3 && python main_V2.py --gpus=2 --N=5 --M=5 --save=CIFAR10_M5N5.V9_prebinact_tune_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0 --cor_loss_weight=0 --pretrained=../CIFAR_10/CIFAR10_M5N5.V7_binact_pretrianed_91.83%/model_best.pth.tar
py3 && python main_V2.py --gpus=3 --N=3 --M=3 --save=CIFAR10_M3N3.V9_prebinact_tune_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0 --cor_loss_weight=0 --pretrained=../CIFAR_10/CIFAR10_M3N3.V7_binact_pretrianed_90.99%/model_best.pth.tar
py3 && python main_V2.py --gpus=6 --N=2 --M=2 --save=CIFAR10_M2N2.V9_prebinact_tune_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0 --cor_loss_weight=0 --pretrained=../CIFAR_10/CIFAR10_M2N2.V7_binact_pretrianed:90.90%/model_best.pth.tar

#Jupyter
py3 && python main_V2.py --gpus=3 --N=5 --M=5 --save=CIFAR10_M5N5.V9_prebinact_tune_alpha_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0.01 --cor_loss_weight=0 --pretrained=../CIFAR_10/CIFAR10_M5N5.V7_binact_pretrianed_91.83%/model_best.pth.tar
#Uranus
py27 && python main_V2.py --gpus=1 --N=3 --M=3 --save=CIFAR10_M3N3.V9_prebinact_tune_alpha_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0.01 --cor_loss_weight=0 --pretrained=../CIFAR_10/CIFAR10_M3N3.V7_binact_pretrianed_90.99%/model_best.pth.tar
py27 && python main_V2.py --gpus=3 --N=2 --M=2 --save=CIFAR10_M2N2.V9_prebinact_tune_alpha_ --arch=vgg_like_multilayer --shift_loss_weight=0 --alpha_loss_weight=0.01 --cor_loss_weight=0 --pretrained=../CIFAR_10/CIFAR10_M2N2.V7_binact_pretrianed:90.90%/model_best.pth.tar
