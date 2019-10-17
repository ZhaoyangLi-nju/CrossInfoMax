from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torchvision import models
from torch.nn import init

# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
#         self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
#         self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
#         self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
#         self.l1 = nn.Linear(512*20*20, 64)
#
#         self.b1 = nn.BatchNorm2d(128)
#         self.b2 = nn.BatchNorm2d(256)
#         self.b3 = nn.BatchNorm2d(512)
#
#     def forward(self, x):
#         h = F.relu(self.c0(x))
#         features = F.relu(self.b1(self.c1(h)))
#         h = F.relu(self.b2(self.c2(features)))
#         h = F.relu(self.b3(self.c3(h)))
#         encoded = self.l1(h.view(x.shape[0], -1))
#         return encoded, features

class Conc_Up_Residual_bottleneck(nn.Module):

    def __init__(self, dim_in, dim_out, stride=1, norm=nn.BatchNorm2d, conc_feat=True):
        super(Conc_Up_Residual_bottleneck, self).__init__()

        self.smooth = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1,
                      padding=1, bias=False),
            norm(dim_out))

        if conc_feat:
            dim_in = dim_out * 2
        else:
            dim_in = dim_out

        dim_med = int(dim_out / 2)
        self.conv1 = nn.Conv2d(dim_in, dim_med, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.norm1 = norm(dim_med)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(dim_med, dim_med)
        self.norm2 = norm(dim_med)
        self.conv3 = nn.Conv2d(dim_med, dim_out, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.norm3 = norm(dim_out)

    def forward(self, x, y=None):

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.smooth(x)
        residual = self.residual_conv(x)

        if y is not None:
            x = torch.cat((x, y), 1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        x += residual

        return self.relu(x)


class Encoder(nn.Module):

    def __init__(self, encoder='resnet18', pretrained='imagenet',in_channel=1):
        super(Encoder, self).__init__()
        # if pretrained == 'imagenet' or pretrained == 'place':
        #     is_pretrained = True
        # else:
        is_pretrained = False

        # if pretrained == 'place':
        #     resnet = models.__dict__[encoder](num_classes=365)
        #     load_path = '/home/dudapeng/workspace/pretrained/place/' + encoder + '_places365.pth'
        #     checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        #     state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        #     resnet.load_state_dict(state_dict)
        #     print('place {0} loaded....'.format(encoder))
        # else:
        #     resnet = models.__dict__[encoder](pretrained=is_pretrained)
        #     print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
        resnet = models.__dict__[encoder](pretrained=False)


        self.conv1 = resnet.conv1
        # self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(512*2*2, 64)
        )

        dims = [32, 64, 128, 256, 512, 1024, 2048]
        norm = nn.InstanceNorm2d
        self.up1 = Conc_Up_Residual_bottleneck(dims[4], dims[3], norm=norm)
        self.up2 = Conc_Up_Residual_bottleneck(dims[3], dims[2], norm=norm)
        self.up3 = Conc_Up_Residual_bottleneck(dims[2], dims[1], norm=norm)
        self.up4 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm)
        self.up5 = Conc_Up_Residual_bottleneck(dims[1], dims[1], norm=norm, conc_feat=False)

        self.up_l_14 = nn.Sequential(
            nn.Conv2d(dims[3], 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.up_l_28 = nn.Sequential(
            nn.Conv2d(dims[2], 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_l_56 = nn.Sequential(
            nn.Conv2d(dims[1], 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_l_112 = nn.Sequential(
            nn.Conv2d(dims[1], 1, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_l_224 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 1, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        self.up_ab_14 = nn.Sequential(
            nn.Conv2d(dims[3], 2, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.up_ab_28 = nn.Sequential(
            nn.Conv2d(dims[2], 2, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_ab_56 = nn.Sequential(
            nn.Conv2d(dims[1], 2, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_ab_112 = nn.Sequential(
            nn.Conv2d(dims[1], 2, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_ab_224 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 2, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        if not is_pretrained:
            init_weights(self, 'normal')

    def forward(self, x, no_grad=False):
        out = {}

        if no_grad:
            with torch.no_grad():
                layer_0 = self.relu(self.bn1(self.conv1(x)))
                layer_1 = self.maxpool(self.layer1(layer_0))
                layer_2 = self.layer2(layer_1)
                out['feat_128'] = layer_2
                layer_3 = self.layer3(layer_2)
                out['feat_256'] = layer_3
                layer_4 = self.layer4(layer_3)
                out['feat_512'] = layer_4
                out['z'] = self.fc(layer_4)
        else:
            layer_0 = self.relu(self.bn1(self.conv1(x)))
            layer_1 = self.maxpool(self.layer1(layer_0))
            layer_2 = self.layer2(layer_1)
            out['feat_128'] = layer_2
            layer_3 = self.layer3(layer_2)
            out['feat_256'] = layer_3
            layer_4 = self.layer4(layer_3)
            out['feat_512'] = layer_4
            # print(layer_4.size())
            out['z'] = self.fc(layer_4)



            L_gen = []
            AB_gen = []


            up1 = self.up1(layer_4, layer_3)
            up2 = self.up2(up1, layer_2)
            up3 = self.up3(up2, layer_1)
            up4 = self.up4(up3, layer_0)
            up5 = self.up5(up4)

            l_14 = self.up_l_14(up1)
            l_28 = self.up_l_28(up2)
            l_56 = self.up_l_56(up3)
            l_112 = self.up_l_112(up4)
            l_224 = self.up_l_224(up5)

            ab_14 = self.up_ab_14(up1)
            ab_28 = self.up_ab_28(up2)
            ab_56 = self.up_ab_56(up3)
            ab_112 = self.up_ab_112(up4)
            ab_224 = self.up_ab_224(up5)


            L_gen.append(l_224)
            L_gen.append(l_112)
            L_gen.append(l_56)
            L_gen.append(l_28)
            L_gen.append(l_14)

            AB_gen.append(ab_224)
            AB_gen.append(ab_112)
            AB_gen.append(ab_56)
            AB_gen.append(ab_28)
            AB_gen.append(ab_14)

            out['L'] = L_gen
            out['AB'] = AB_gen


        return out


class Contrastive_CrossModal_Conc(nn.Module):

    def __init__(self, cfg, alpha=0.5, beta=1.0, gamma=0.05, feat_channel=128, feat_size=8, device=None):
        super(Contrastive_CrossModal_Conc, self).__init__()

        self.cfg = cfg
        self.device = device
        self.encoder_rgb = Encoder(pretrained='')
        self.evaluator_rgb = Evaluator(cfg.NUM_CLASSES)
        # self.avg_pool = nn.AvgPool2d(7, 1)
        self.avg_pool = nn.AvgPool2d(2, 1)

        # self.z = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(1024 * 7 * 7, 64)
        # )

        self.prior_d_cross = PriorDiscriminator(64)
        self.local_d_inner_rgb = LocalDiscriminator(feat_channel + 64)
        self.local_l_cross = LocalDiscriminator(2)  # + 128
        self.local_ab_cross = LocalDiscriminator(4)  # + 128


        # self.cls_criterion = torch.nn.CrossEntropyLoss(cfg.CLASS_WEIGHTS_TRAIN)
        self.cls_criterion = torch.nn.CrossEntropyLoss()

        self.pixel_criterion = torch.nn.L1Loss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.feat_size = feat_size

        init_weights(self, 'normal')
        #
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')
        # init_weights(self, 'normal')

    def forward(self, rgb, l=None, ab=None, label=None, class_only=False):
        res_dict = {}
        if class_only:
            # shortcut to encode one image and evaluate classifier
            result_rgb = self.encoder_rgb(rgb, no_grad=True)
            avg_rgb = self.avg_pool(result_rgb['feat_512'])
            lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)
            res_dict['class_rgb'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
            return res_dict
        # l,ab = torch.split(rgb[1],[1,2],dim=1) 

        # depth={}
        # for i in range(self.cfg.MULTI_SCALE_NUM):
        #     l,ab = torch.split(rgb[i],[1,2],dim=1)
        #     depth.append(ab)
        result_rgb = self.encoder_rgb(rgb)
        avg_rgb = self.avg_pool(result_rgb['feat_512'])
        lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)

        z_rgb = result_rgb['z']
        f_rgb = result_rgb['feat_128']

        z_rgb_exp = z_rgb.unsqueeze(-1).unsqueeze(-1)
        z_rgb_exp = z_rgb_exp.expand(-1, -1, self.feat_size, self.feat_size)

        prior = torch.rand_like(z_rgb)

        term_a = torch.log(self.prior_d_cross(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d_cross(z_rgb)).mean()
        PRIOR_RGB = - (term_a + term_b) * self.gamma
        inner_neg_rgb = torch.cat((f_rgb[1:], f_rgb[0].unsqueeze(0)), dim=0)
        # print(f_rgb.size())
        # print(z_rgb_exp.size())
        inner_y_M_RGB = torch.cat((f_rgb, z_rgb_exp), dim=1)


        inner_y_M_prime_RGB = torch.cat((inner_neg_rgb, z_rgb_exp), dim=1)

        Ej = -F.softplus(-self.local_d_inner_rgb(inner_y_M_RGB)).mean()
        Em = F.softplus(self.local_d_inner_rgb(inner_y_M_prime_RGB)).mean()
        LOCAL_RGB = (Em - Ej) * self.beta

        LOCAL_cross_ab = torch.zeros(1).to(self.device)
        LOCAL_cross_l = torch.zeros(1).to(self.device)

        for i, (L_gen,AB_gen,L_label, AB_label) in enumerate(zip(result_rgb['L'],result_rgb['AB'],l,ab)):

            if i + 1 > self.cfg.MULTI_SCALE_NUM:
                break

            cross_pos = torch.cat((L_gen, L_label), 1)
            cross_neg = torch.cat((L_gen, torch.cat((L_label[1:], L_label[0].unsqueeze(0)), dim=0)), 1)
            Ej = -F.softplus(-self.local_l_cross(cross_pos)).mean()
            Em = F.softplus(self.local_l_cross(cross_neg)).mean()
            LOCAL_cross_l += (Em - Ej) * self.beta

            cross_pos = torch.cat((AB_gen, AB_label), 1)
            cross_neg = torch.cat((AB_gen, torch.cat((AB_label[1:], AB_label[0].unsqueeze(0)), dim=0)), 1)
            Ej = -F.softplus(-self.local_ab_cross(cross_pos)).mean()
            Em = F.softplus(self.local_ab_cross(cross_neg)).mean()
            LOCAL_cross_ab += (Em - Ej) * self.beta
        # cls_loss = self.cls_criterion(lgt_glb_mlp_rgb, label) + self.cls_criterion(lgt_glb_lin_rgb, label)

        # pixel_loss = self.pixel_criterion(result_rgb['ms_gen'][0], depth[0])

        # return {'cls_loss': cls_loss, 'pixel_loss': pixel_loss, 'ms_gen': result_rgb['ms_gen']}
        return {'gen_l_loss': sum(LOCAL_cross_l),'gen_ab_loss': sum(LOCAL_cross_ab), 'local_rgb_loss': LOCAL_RGB, 
                'prior_rgb_loss': PRIOR_RGB, 'l_gen': result_rgb['L'], 'ab_gen': result_rgb['AB']}


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.c0 = nn.Conv2d(in_channel, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 10 * 10 + 128, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.c0 = nn.Conv2d(in_channel, 64, kernel_size=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=1)
        self.c4 = nn.Conv2d(512, 1, kernel_size=1)

        init_weights(self, 'normal')

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        return self.c4(h)


class PriorDiscriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.l0 = nn.Linear(in_channel, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

        init_weights(self, 'normal')

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz

class Evaluator(nn.Module):
    def __init__(self, n_classes):
        super(Evaluator, self).__init__()
        self.n_classes = n_classes
        self.block_glb_mlp = \
            MLPClassifier(512, self.n_classes, n_hidden=1024, p=0.2)
        self.block_glb_lin = \
            MLPClassifier(512, self.n_classes, n_hidden=None, p=0.0)

    def forward(self, ftr_1):
        '''
        Input:
          ftr_1 : features at 1x1 layer
        Output:
          lgt_glb_mlp: class logits from global features
          lgt_glb_lin: class logits from global features
        '''
        # collect features to feed into classifiers
        # - always detach() -- send no grad into encoder!
        h_top_cls = flatten(ftr_1).detach()
        # h_top_cls = flatten(ftr_1)
        # compute predictions
        lgt_glb_mlp = self.block_glb_mlp(h_top_cls)
        lgt_glb_lin = self.block_glb_lin(h_top_cls)
        return lgt_glb_mlp, lgt_glb_lin


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super(MLPClassifier, self).__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if n_hidden is None:
            # use linear classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            # use simple MLP classifier
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        logits = self.block_forward(x)
        return logits

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)

def flatten(x):
    return x.reshape(x.size(0), -1)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class UpsampleBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, scale=2,
                 mode='bilinear', upsample=True):
        super(UpsampleBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        if upsample:
            if inplanes != planes:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 3, 1

            self.upsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                          padding=padding, bias=False),
                norm(planes))
        else:
            self.upsample = None

        self.scale = scale
        self.mode = mode

    def forward(self, x):

        if self.upsample is not None:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
            residual = self.upsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]