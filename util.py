import torch.nn as nn
import numpy
import torch
import sys
sys.path.insert(0, './models')
import binary_connect_network_real
class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # # self.target_modules[index].grad.data = \
            # #         self.target_modules[index].grad.data.mul(m)
            # m = m.mul(self.target_modules[index].grad.data)
            # m_add = weight.sign().mul(self.target_modules[index].grad.data)
            # m_add = m_add.sum(3, keepdim=True)\
            #         .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            # m_add = m_add.mul(weight.sign())
            # self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)

class BinOp_V2():
    def __init__(self, model, M, N, pretrained_model=None):
        # count the number of Conv2d and Linear
        count_targets = 0
        count_alpha = 0
        self.N = N
        self.M = M
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-2
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1).astype('int').tolist()
        self.num_of_params = len(self.bin_range)

        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        #################### for initlization #########################################
        self.target_bias = []
        self.target_bn = []
        #############################################################
        self.pretrained_model = pretrained_model
        index = -1
        self.model=model
        # initlize every layer, model.weights model.alpha and model.shift

        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)
                    # print(m.weight.data.size())
                    self.target_bias.append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                # print("init bn",m)
                self.target_bn.append(m)
        # self.target_alpha=[]
        # for name, data in model.named_parameters():
        #     if('alpha' in name):
        #         self.target_alpha.append(data)
        self.target_shift = []
        for name, data in model.named_parameters():
            if('shift' in name):
                self.target_shift.append(data)


        self.target_beta = []
        for name, data in model.named_parameters():
            if ('beta' in name):
                self.target_beta.append(data)
            # print(name)
        # print(len(self.target_alpha))
        # print(self.num_of_params)
        # assert len(self.target_alpha)==self.num_of_params
        
    def binarization_pre(self):
        self.meancenterConvParams()
        self.clampConvParams()
        if self.pretrained_model=='None':
            print('====> not using any pretrianed model ')
        else:
            print('====> using pretrianed model ' + self.pretrained_model)
            self.pretrained(self.pretrained_model)
        self.save_params()
        self.binarizeConvParams()

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def pretrained(self, pretrained_model):
        pretrained_count_targets = 0
        state_dict = torch.load(pretrained_model, map_location=lambda storage, loc: storage)['state_dict']
        pretrained_model = binary_connect_network_real.vgg_like_real_binact(N=self.N, M=self.M)
        pretrained_model = torch.nn.DataParallel(pretrained_model, device_ids=[0])
        pretrained_model.load_state_dict(state_dict)
        for m in pretrained_model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                pretrained_count_targets = pretrained_count_targets + 1
        assert self.num_of_params==(pretrained_count_targets-2)*self.M
        pretrained_bin_range = numpy.linspace(1, pretrained_count_targets - 2, 
                            pretrained_count_targets - 2).astype('int').tolist()
        pretrained_model=pretrained_model.cpu().cuda()
        index = -1
        count = 0
        bncount = 0
        shift_count = 0
        beta_count = 0
        if_beta = True
        for m in pretrained_model.modules():

            if isinstance(m, nn.ParameterList):
                if if_beta:
                    for i in range(self.N):
                        print("beta ", i, m[i].data),
                        print("coping to ", beta_count)
                        self.target_beta[beta_count].data.copy_(m[i].data)
                        self.target_beta[beta_count].requires_grad = False
                        beta_count = beta_count + 1
                else:
                    for i in range(self.N):
                        print("shift", i, m[i].data),
                        print("coping to ", shift_count)
                        self.target_shift[shift_count].data.copy_(m[i].data)
                        self.target_shift[shift_count].requires_grad = False
                        shift_count = shift_count + 1
                if_beta = not if_beta

            if isinstance(m,nn.BatchNorm2d):
                # print(m)
                # print(self.target_bn[bncount])
                self.target_bn[bncount].bias.data.copy_(m.bias.data)
                self.target_bn[bncount].running_mean.data.copy_(m.running_mean.data)
                self.target_bn[bncount].weight.data.copy_(m.weight.data)
                self.target_bn[bncount].running_var.data.copy_(m.running_var.data)
                self.target_bn[bncount].num_batches_tracked.data.copy_(m.num_batches_tracked.data)
                bncount = bncount+1
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index==0:
                    for m_bin in self.model.modules():
                        if isinstance(m_bin, nn.Conv2d) or isinstance(m_bin, nn.Linear):
                            m_bin.weight.data = m.weight.data.clone()
                            m_bin.bias.data = m.bias.data.clone()
                            break
                elif index not in pretrained_bin_range:
                    tmpcou=0
                    for m_bin in self.model.modules():
                        if isinstance(m_bin, nn.Conv2d) or isinstance(m_bin, nn.Linear):
                            if(tmpcou==self.num_of_params+1):
                                m_bin.bias.data = m.bias.data.clone()
                                m_bin.weight.data = m.weight.data.clone()
                                break
                            tmpcou=tmpcou+1
                elif index in pretrained_bin_range:
                    tmp = m.weight.data.clone()
                    tmpbias = m.bias.data
                    # print(index)
                    # print(count)
                    self.target_modules[count].data = tmp
                    self.target_modules[count].data = self.binarizeConvParams_one(self.target_modules[count].data)
                    self.target_bias[count].data = tmpbias.clone()/self.M
                    for j in range(1, self.M):
                        new_tensor = m.weight.data.clone()
                        for k in range(j):
                            # print("k=",k)
                            # print(new_tensor.size(),self.target_modules[count+k].data.size())
                            new_tensor = new_tensor - self.target_modules[count+k].data.clone()
                        self.target_bias[count+j].data = tmpbias.clone()/self.M
                        self.target_modules[count+j].data = self.binarizeConvParams_one(new_tensor)
                    count =count + self.M
        del pretrained_model
        self.save_params()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams_one(self, layer):
        n = layer[0].nelement()
        s = layer.size()
        if len(s) == 4:
            m = layer.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
        elif len(s) == 2:
            m = layer.norm(1, 1, keepdim=True).div(n)
        else:
            print(s)
            assert 1==2
        # m = layer.norm(1, 3, keepdim=True)\
        #         .sum(2, keepdim=True).sum(1, keepdim=True).div(n)

        # print("m size is ",m.size())
        # print("s is ",s)
        # print(m)
        return layer.sign().mul(m.expand(s))
    
    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))
            #########################V2################################################
            # #do not need to mul(m.expand(s))
            #########################V2################################################

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    # update grad ver.1
    def updateBinaryGradWeight(self): 
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            #########################V1################################################
            ### alpha for the weights
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            #########################V1################################################

            #########################V2################################################
            # alpha for the weights
            # m = self.target_alpha[index].data.data.expand(s)
            #########################V2################################################


            m[weight.lt(-1.0)] = 0 # Derivatives for sign(W)/W
            m[weight.gt(1.0)] = 0  # 

            m = m.mul(self.target_modules[index].grad.data)

            #########################V1################################################
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            # loss multiply the sign
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
            # the loss of average Weights, it is : sum(sign(W)*loss)*sign(W)/n
            #########################V1################################################
            # self.target_modules[index].grad.data = m.mul(1.0-1.0/s[1]).mul(n)
            # mul(1.0-1.0/s[1]) is the scale for meancenterConvParams()
            # .mul(n) is used to mitigate the effect of weight decay.
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)

    # update grad ver.2
    def updateBinaryGradWeight_v2(self): 
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            #########################V1################################################
            ### alpha for the weights
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
            #########################V1################################################

            #########################V2################################################
            # alpha for the weights
            # m = self.target_alpha[index].data.data.expand(s)
            #########################V2################################################


            m[weight.lt(-1.0)] = 0 # Derivatives for sign(W)/W
            m[weight.gt(1.0)] = 0  # 

            m = m.mul(self.target_modules[index].grad.data)

            #########################V1################################################
            # m_add = weight.sign().mul(self.target_modules[index].grad.data)
            # # loss multiply the sign
            # if len(s) == 4:
            #     m_add = m_add.sum(3, keepdim=True)\
            #             .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            # elif len(s) == 2:
            #     m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
            # m_add = m_add.mul(weight.sign())
            # self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

            # the loss of average Weights, it is : sum(sign(W)*loss)*sign(W)/n
            #########################V1################################################
            self.target_modules[index].grad.data = m.mul(1.0-1.0/s[1]).mul(n)
            # mul(1.0-1.0/s[1]) is the scale for meancenterConvParams()
            # .mul(n) is used to mitigate the effect of weight decay.
            self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)