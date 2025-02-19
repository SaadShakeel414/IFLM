from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import Parameter
import math


# Support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']

class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        """
    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zero_(self.bias)

    def forward(self, x):
        if self.device_id == None:
            out = F.linear(x, self.weight, self.bias)
        else:
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.30, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
      
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1) 
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

class SphereFace(nn.Module):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, device_id, m = 4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cos_theta = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', m = ' + str(self.m) + ')'


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class Am_softmax(nn.Module):
    r"""Implement of Am_softmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
    """
    def __init__(self, in_features, out_features, device_id, m = 0.35, s = 30.0):
        super(Am_softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.device_id = device_id

        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initialize kernel
 
    def forward(self, embbedings, label):
        if self.device_id == None:
            kernel_norm = l2_norm(self.kernel, axis = 0)
            cos_theta = torch.mm(embbedings, kernel_norm)
        else:
            x = embbedings
            sub_kernels = torch.chunk(self.kernel, len(self.device_id), dim=1)
            temp_x = x.cuda(self.device_id[0])
            kernel_norm = l2_norm(sub_kernels[0], axis = 0).cuda(self.device_id[0])
            cos_theta = torch.mm(temp_x, kernel_norm)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                kernel_norm = l2_norm(sub_kernels[i], axis = 0).cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, torch.mm(temp_x, kernel_norm).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output
    
class CurricularFace(nn.Module):
    """Implementation for "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition".
    """
    def __init__(self, feat_dim, num_class, device_id, m = 0.5, s = 64):
        super(CurricularFace, self).__init__()
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(feat_dim, num_class))
        self.register_buffer('t', torch.zeros(1))
        self.device_id = device_id
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, feats, labels):
        kernel_norm = F.normalize(self.kernel, dim=0).cuda(self.device_id[0])
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, feats.size(0)), labels].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m #cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask].cuda(self.device_id[0])
        with torch.no_grad():
            self.t = (target_logit.mean() * 0.01 + (1 - 0.01) * self.t).cuda(self.device_id[0])
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, labels.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output

# class MagFace(nn.Module):
#     """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
#     """
#     def __init__(self, feat_dim, num_class, device_id, margin_am=0.0, scale=64.0, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8, lamda=20):
#         super(MagFace, self).__init__()
#         self.weight = Parameter(torch.Tensor(feat_dim, num_class))
#         self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#         self.margin_am = margin_am
#         self.scale = scale
#         self.l_a = l_a
#         self.u_a = u_a
#         self.l_margin = l_margin
#         self.u_margin = u_margin
#         self.lamda = lamda
#         self.device_id = device_id
#
#     def calc_margin(self, x):
#         margin = (self.u_margin-self.l_margin) / \
#                  (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
#         return margin
#
#     def forward(self, feats, labels):
#         x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(self.l_a, self.u_a)# l2 norm
#         ada_margin = self.calc_margin(x_norm)
#         cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
#         loss_g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
#         kernel_norm = F.normalize(self.weight, dim=0).cuda(self.device_id[0])
#         feats = F.normalize(feats)
#         cos_theta = torch.mm(feats, kernel_norm)
#         cos_theta = cos_theta.clamp(-1, 1)
#         sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
#         cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
#         # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
#         # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
#         min_cos_theta = torch.cos(math.pi - ada_margin)
#         cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta-self.margin_am)
#         index = torch.zeros_like(cos_theta)
#         index.scatter_(1, labels.data.view(-1, 1), 1)
#         index = index.byte()
#         output = cos_theta * 1.0
#         output[index] = cos_theta_m[index]
#         output *= self.scale
#
#         return output
#        return output, self.lamda*loss_g

class MagFace(nn.Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, feat_dim, num_class, device_id, margin_am=0.0, scale=32, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8,
                 lamda=2):
        super(MagFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_am = margin_am
        self.scale = scale
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lamda = lamda
        self.device_id = device_id

    def calc_margin(self, x):
        margin = (self.u_margin - self.l_margin) / \
                 (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
        return margin

    def forward(self, feats, labels):
        feats = feats.to(self.device_id[0])
        # self.weight = self.weight.to(self.device_id[0])
        x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(self.l_a, self.u_a)  # l2 norm
        ada_margin = self.calc_margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        loss_g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        kernel_norm = F.normalize(self.weight, dim=0)
        feats = F.normalize(feats)
        cos_theta = torch.mm(feats, kernel_norm).to(self.device_id[0])
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin).to(self.device_id[0])
        cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta - self.margin_am)
        index = torch.zeros_like(cos_theta)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output, self.lamda * loss_g


class ElasticArcFace(nn.Module):
    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50,std=0.0125,plus=False):
        super(ElasticArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.plus=plus
        self.device_id = device_id
    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0).cuda(self.device_id[0])
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device) # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta
    
    
class ElasticCosFace(nn.Module):
    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35,std=0.0125, plus=False):
        super(ElasticCosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.plus=plus
        self.device_id = device_id

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0).cuda(self.device_id[0])
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)  # Fast converge .clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret


class AdaFace(Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=10575,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=1.0,
                 ):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        # print('\n\AdaFace with the following property')
        # print('self.m', self.m)
        # print('self.h', self.h)
        # print('self.s', self.s)
        # print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel,axis=0).to(embbedings.device)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean.to(mean.device)
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std.to(std.device)

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class MagFace(Module):
    """Implementation for "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, feat_dim, num_class, margin_am=0.0, scale=32, l_a=10, u_a=110, l_margin=0.45, u_margin=0.8,
                 lamda=20):
        super(MagFace, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin_am = margin_am
        self.scale = scale
        self.l_a = l_a
        self.u_a = u_a
        self.l_margin = l_margin
        self.u_margin = u_margin
        self.lamda = lamda

    def calc_margin(self, x):
        margin = (self.u_margin - self.l_margin) / \
                 (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
        return margin

    def forward(self, feats, labels):
        device = feats.device
        self.weight.data = self.weight.data.to(device)
        x_norm = torch.norm(feats, dim=1, keepdim=True).clamp(self.l_a, self.u_a)  # l2 norm
        ada_margin = self.calc_margin(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        loss_g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        kernel_norm = F.normalize(self.weight, dim=0).to(device)
        feats = F.normalize(feats).to(device)
        cos_theta = torch.mm(feats, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        # 0 <= theta + m <= pi, ==> -m <= theta <= pi-m
        # because 0<=theta<=pi, so, we just have to keep theta <= pi-m, that is cos_theta >= cos(pi-m)
        min_cos_theta = torch.cos(math.pi - ada_margin)
        cos_theta_m = torch.where(cos_theta > min_cos_theta, cos_theta_m, cos_theta - self.margin_am)
        index = torch.zeros_like(cos_theta).to(device)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.scale
        return output, self.lamda * loss_g

######################################################SphereFace2#####################################################
def softplus_wrapping(raw_feats):
    '''
    Args:
        raw_feats: (B, L) or (B, C, L)
    Returns:
        out: (B, L) or (B, C, L)
    '''
    mags = F.softplus(raw_feats[..., :1], beta=1)
    feats = mags * F.normalize(raw_feats[..., 1:], dim=-1)
    return feats

def get_wrapping(mode):
    if mode == 'cartesian':
        f_wrapping = lambda raw_feats: raw_feats
    elif mode == 'polarlike':
        f_wrapping = softplus_wrapping
    else:
        raise ValueError('mode must be `cartesian` or `polarlike`')
    return f_wrapping


def weighted_average(feats, weights=None, pre_norm=False, post_norm=True):
    '''
    Args:
        feats: (B, C, L) or (C, L)
        B: batch size
        C: the number of observations from the same subject
        L: feature dimension
        weights: None or (C, ), weights for each observation
        pre_norm: normalize the features before fusion
        post_norm: normalize the fused feature
    Returns:
        fuse_feat: (B, L) or (L, )
    '''
    if weights is None:
        weights = torch.ones(feats.size(-2))
    weights = weights / weights.sum()
    weights = weights.to(feats.device)

    assert weights.ndim == 1
    if feats.ndim == 2:
        weights = weights.view(-1, 1)
    elif feats.ndim == 3:
        weights = weights.view(1, -1, 1)
    else:
        raise ValueError("Invalid shape for `feats`")

    if pre_norm:
        feats = F.normalize(feats, dim=-1)
    fuse_feat = torch.sum(weights * feats, dim=-2)
    if post_norm:
        fuse_feat = F.normalize(fuse_feat, dim=-1)

    return fuse_feat

def get_fusing(pre_norm, post_norm):
    f_fusing = lambda feats, weights=None: \
        weighted_average(feats, weights, pre_norm, post_norm)
    return f_fusing


def generalized_inner_product(feats_1, feats_2, b_theta=0., all_pairs=False):
    '''
    Args:
        feats_1: (B1, L)
        feats_2: (B2, L)
    Returns:
        scores: (B1, )   if all_pairs is False and B1==B2
                (B1, B2) if all_pairs is True
    '''
    assert feats_1.ndim == feats_2.ndim == 2
    assert feats_1.size(1) == feats_2.size(1)

    mags_1 = torch.norm(feats_1, p=2, dim=1, keepdim=True)
    mags_2 = torch.norm(feats_2, p=2, dim=1, keepdim=True)
    if all_pairs:
        scores = feats_1.mm(feats_2.t()) - (b_theta * mags_1).mm(mags_2.t())
    else:
        scores = (feats_1 * feats_2).sum(dim=1) - (b_theta * mags_1 * mags_2).flatten()
    return scores

def get_scoring(b_theta):
    f_scoring = lambda feats_1, feats_2, all_pairs=False: \
        generalized_inner_product(feats_1, feats_2, b_theta, all_pairs)
    return f_scoring


class SphereFace2(nn.Module):
    """ reference: <SphereFace2: Binary Classification is All You Need
                    for Deep Face Recognition> SphereFace2-C
    """
    def __init__(self, feat_dim, subj_num,
            alpha=0.7, r=40., m=0.4, t=3., lw=50.):
        super().__init__()
        #
        self.f_wrapping = get_wrapping(mode='cartesian')
        self.f_fusing = get_fusing(pre_norm=False, post_norm=True)
        self.f_scoring = get_scoring(b_theta=0.)

        self.feat_dim = feat_dim
        self.subj_num = subj_num
        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(subj_num, feat_dim))
        nn.init.xavier_normal_(self.w)

        # init bias
        z = alpha / ((1. - alpha) * (subj_num - 1.))
        ay = r * (2. * 0.5**t - 1. - m)
        ai = r * (2. * 0.5**t - 1. + m)

        temp = (1. - z)**2 + 4. * z * math.exp(ay - ai)
        b = math.log(2. * z) - ai - math.log(1. - z +  math.sqrt(temp))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=-1)

        #delta theta with margin
        x = self.f_wrapping(x)
        x = self.f_fusing(x[:, None, :])
        cos_theta = self.f_scoring(x, self.w, all_pairs=True)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.)
        with torch.no_grad():
            g_cos_theta = 2. * ((cos_theta + 1.) / 2.).pow(self.t) - 1.
            g_cos_theta = g_cos_theta - self.m * (2. * one_hot - 1.)
            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.alpha * one_hot + (1. - self.alpha) * (1. - one_hot)
        weight = self.lw * self.subj_num / self.r * weight
        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)

        return loss