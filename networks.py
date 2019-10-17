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

class Encoder(nn.Module):

    def __init__(self, encoder='resnet18', pretrained='imagenet'):
        super(Encoder, self).__init__()
        if pretrained == 'imagenet' or pretrained == 'place':
            is_pretrained = True
        else:
            is_pretrained = False

        if pretrained == 'place':
            resnet = models.__dict__[encoder](num_classes=365)
            load_path = '/home/dudapeng/workspace/pretrained/place/' + encoder + '_places365.pth'
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place {0} loaded....'.format(encoder))
        else:
            resnet = models.__dict__[encoder](pretrained=is_pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(512 * 7 * 7, 64)
        )

        dims = [32, 64, 128, 256, 512, 1024, 2048]
        norm = nn.InstanceNorm2d
        self.up1 = Upsample_Interpolate(dims[4], dims[1], norm=norm)
        self.up2 = Upsample_Interpolate(dims[1], dims[1], norm=norm)
        self.up3 = Upsample_Interpolate(dims[1], dims[1], norm=norm)
        self.up4 = Upsample_Interpolate(dims[1], dims[1], norm=norm)
        self.up5 = Upsample_Interpolate(dims[1], dims[1], norm=norm)
        self.lat1 = conv_norm_relu(dims[3], dims[1], kernel_size=1, padding=0, norm=norm)
        self.lat2 = conv_norm_relu(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
        self.lat3 = conv_norm_relu(dims[1], dims[1], kernel_size=1, padding=0, norm=norm)
        self.lat4 = conv_norm_relu(dims[1], dims[1], kernel_size=1, padding=0, norm=norm)

        self.up_image_14 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

        self.up_image_28 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_image_56 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        self.up_image_112 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )
        self.up_image_224 = nn.Sequential(
            # conv_norm_relu(dims[1], dims[1], norm=norm),
            nn.Conv2d(dims[1], 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        if not is_pretrained:
            init_weights(self, 'normal')

    def forward(self, x, no_grad=False):
        out = {}

        if no_grad:
            with torch.no_grad():
                out['0'] = self.relu(self.bn1(self.conv1(x)))
                pool = self.maxpool(out['0'])
                out['1'] = self.layer1(pool)
                out['2'] = self.layer2(out['1'])
                out['feat_128'] = out['2']
                out['3'] = self.layer3(out['2'])
                out['feat_256'] = out['3']
                out['4'] = self.layer4(out['3'])
                out['feat_512'] = out['4']
                out['z'] = self.fc(out['4'])
        else:
            out['0'] = self.relu(self.bn1(self.conv1(x)))
            pool = self.maxpool(out['0'])
            out['1'] = self.layer1(pool)
            out['2'] = self.layer2(out['1'])
            out['feat_128'] = out['2']
            out['3'] = self.layer3(out['2'])
            out['feat_256'] = out['3']
            out['4'] = self.layer4(out['3'])
            out['feat_512'] = out['4']
            out['z'] = self.fc(out['4'])

            ms_gen = []

            skip_1 = self.lat1(out['3'])
            skip_2 = self.lat2(out['2'])
            skip_3 = self.lat3(out['1'])
            skip_4 = self.lat4(out['0'])

            up1 = self.up1(out['4'])
            up2 = self.up2(up1 + skip_1)
            up3 = self.up3(up2 + skip_2)
            up4 = self.up4(up3 + skip_3)
            up5 = self.up5(up4 + skip_4)
            gen_14 = self.up_image_14(up1)
            gen_28 = self.up_image_28(up2)
            gen_56 = self.up_image_56(up3)
            gen_112 = self.up_image_112(up4)
            gen_224 = self.up_image_224(up5)

            ms_gen.append(gen_224)
            ms_gen.append(gen_112)
            ms_gen.append(gen_56)
            ms_gen.append(gen_28)
            ms_gen.append(gen_14)

            out['ms_gen'] = ms_gen

        return out

#

# class Contrastive_CrossModal(nn.Module):
#
#     def __init__(self, cfg):
#         super(Contrastive_CrossModal, self).__init__()
#         self.encoder_rgb = Encoder(pretrained='')
#         self.evaluator_rgb = Evaluator(cfg.NUM_CLASSES)
#         self.encoder_depth = Encoder(pretrained='')
#         self.evaluator_depth = Evaluator(cfg.NUM_CLASSES)
#         self.avg_pool = nn.AvgPool2d(7, 1)
#
#         init_weights(self, 'normal')
#
#     def forward(self, rgb, depth, class_only=False):
#         res_dict = {}
#         if class_only:
#             # shortcut to encode one image and evaluate classifier
#             _, _, _, f_rgb_512 = self.encoder_rgb(rgb, no_grad=True)
#             _, _, _, f_depth_512 = self.encoder_depth(depth, no_grad=True)
#             avg_rgb = self.avg_pool(f_rgb_512)
#             avg_depth = self.avg_pool(f_depth_512)
#             lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)
#             lgt_glb_mlp_depth, lgt_glb_lin_depth = self.evaluator_depth(avg_depth)
#             res_dict['class_rgb'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
#             res_dict['class_depth'] = [lgt_glb_mlp_depth, lgt_glb_lin_depth]
#             return res_dict
#
#         z_rgb, f_rgb_128, f_rgb_256, f_rgb_512 = self.encoder_rgb(rgb)
#         z_depth, f_depth_128, f_depth_256, f_depth_512 = self.encoder_depth(depth)
#
#         avg_rgb = self.avg_pool(f_rgb_512)
#         avg_depth = self.avg_pool(f_depth_512)
#         lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)
#         lgt_glb_mlp_depth, lgt_glb_lin_depth = self.evaluator_depth(avg_depth)
#
#         res_dict['f_rgb'] = f_rgb_128
#         res_dict['z_rgb'] = z_rgb
#         # res_dict['gen_rgb'] = gen_rgb
#         res_dict['f_depth'] = f_depth_128
#         res_dict['z_depth'] = z_depth
#         # res_dict['gen_depth'] = gen_depth
#         res_dict['class_rgb'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
#         res_dict['class_depth'] = [lgt_glb_mlp_depth, lgt_glb_lin_depth]
#
#         return res_dict

class Contrastive_CrossModal_Conc(nn.Module):

    def __init__(self, cfg):
        super(Contrastive_CrossModal_Conc, self).__init__()
        self.encoder_rgb = Encoder(pretrained='')
        self.evaluator_rgb = Evaluator(cfg.NUM_CLASSES)
        # self.encoder_depth = Encoder(pretrained='')
        # self.evaluator_depth = Evaluator(cfg.NUM_CLASSES)
        self.avg_pool = nn.AvgPool2d(7, 1)
        self.z = nn.Sequential(
            Flatten(),
            nn.Linear(1024 * 7 * 7, 64)
        )

        init_weights(self, 'normal')\


    def forward(self, rgb, depth, label=None, class_only=False):
        res_dict = {}
        if class_only:
            # shortcut to encode one image and evaluate classifier
            result_rgb = self.encoder_rgb(rgb, no_grad=True)
            # result_depth = self.encoder_depth(depth, no_grad=True)
            avg_rgb = self.avg_pool(result_rgb['feat_512'])
            # avg_depth = self.avg_pool(result_depth['feat_512'])
            lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)
            # lgt_glb_mlp_depth, lgt_glb_lin_depth = self.evaluator_depth(avg_depth)
            res_dict['class_rgb'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
            # res_dict['class_depth'] = [lgt_glb_mlp_depth, lgt_glb_lin_depth]
            return res_dict

        result_rgb = self.encoder_rgb(rgb)
        # result_depth = self.encoder_depth(depth)
        # z = self.z(torch.cat((result_rgb['feat_512'], result_depth['feat_512']), 1))

        avg_rgb = self.avg_pool(result_rgb['feat_512'])
        # avg_depth = self.avg_pool(result_depth['feat_512'])
        lgt_glb_mlp_rgb, lgt_glb_lin_rgb = self.evaluator_rgb(avg_rgb)
        # lgt_glb_mlp_depth, lgt_glb_lin_depth = self.evaluator_depth(avg_depth)

        # res_dict['z'] = z
        res_dict['f_rgb'] = result_rgb['feat_128']
        res_dict['z_rgb'] = result_rgb['z']
        res_dict['ms_gen'] = result_rgb['ms_gen']
        # res_dict['f_depth'] = result_depth['feat_128']
        # res_dict['z_depth'] = result_depth['z']
        # res_dict['gen_depth'] = gen_depth
        res_dict['class_rgb'] = [lgt_glb_mlp_rgb, lgt_glb_lin_rgb]
        # res_dict['class_depth'] = [lgt_glb_mlp_depth, lgt_glb_lin_depth]

        return res_dict

# class TrecgNet(nn.Module):
#     def __init__(self, encoder='resnet18', pretrained=False):
#         super(TrecgNet, self).__init__()
#
#         self.encoder = encoder
#         pretrained = pretrained
#         self.dim_noise = 256
#         self.avg_pool_size = 14
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if pretrained == 'imagenet' or pretrained == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if pretrained == 'place':
#             resnet = models.__dict__[encoder](num_classes=365)
#             load_path = '/home/dudapeng/workspace/pretrained/place/' + encoder + '_places365.pth'
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place {0} loaded....'.format(encoder))
#         else:
#             resnet = models.__dict__[encoder](pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.fc = nn.Linear(512*7*7, 128)
#
#         self.build_upsample_layers(dims)
#
#         if pretrained:
#
#             init_weights(self.deconv1, 'normal')
#             init_weights(self.deconv2, 'normal')
#             init_weights(self.deconv3, 'normal')
#             init_weights(self.deconv4, 'normal')
#             init_weights(self.latlayer1, 'normal')
#             init_weights(self.latlayer2, 'normal')
#             init_weights(self.latlayer3, 'normal')
#             init_weights(self.up_image, 'normal')
#
#         elif not pretrained:
#
#             init_weights(self, 'normal')
#
#     def build_upsample_layers(self, dims):
#
#         # norm = nn.BatchNorm2d
#         norm = nn.InstanceNorm2d
#
#         self.deconv1 = UpsampleBasicBlock(dims[4], dims[3], kernel_size=1, padding=0, norm=norm)
#         self.deconv2 = UpsampleBasicBlock(dims[3], dims[2], kernel_size=1, padding=0, norm=norm)
#         self.deconv3 = UpsampleBasicBlock(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
#         self.deconv4 = UpsampleBasicBlock(dims[1], dims[1], kernel_size=3, padding=1, norm=norm)
#         self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
#         self.up_image = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#
#         out = {}
#         # conv1 = self.conv1(x)
#         # out['0'] = self.relu(self.bn1(conv1))
#         out['0'] = self.maxpool(self.relu(self.bn1(self.conv1(x))))
#
#         out['1'] = self.layer1(out['0'])
#         out['2'] = self.layer2(out['1'])
#         out['3'] = self.layer3(out['2'])
#         out['4'] = self.layer4(out['3'])
#
#         feature = out['4']
#         z = self.fc(feature.view(feature.shape[0], -1))
#
#         skip1 = self.latlayer1(out['3'])
#         skip2 = self.latlayer2(out['2'])
#         skip3 = self.latlayer3(out['1'])
#
#         # skip1 = out['3']
#         # skip2 = out['2']
#         # skip3 = out['1']
#
#         upconv1 = self.deconv1(out['4'])
#         upconv2 = self.deconv2(upconv1 + skip1)
#         upconv3 = self.deconv3(upconv2 + skip2)
#         upconv4 = self.deconv4(upconv3 + skip3)
#
#         x = self.up_image(upconv4)
#         return z, feature, x

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


# class GlobalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.m = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#         )
#         self.l0 = nn.Linear(64 * 28 * 28 + 128, 512)
#         self.l1 = nn.Linear(512, 512)
#         self.l2 = nn.Linear(512, 1)
#         init_weights(self, 'normal')
#
#     def forward(self, y, M):
#         # h = F.relu(self.c0(M))
#         # h = self.c1(h)
#         h = self.m(M)
#         h = h.view(y.shape[0], -1)
#         h = torch.cat((y, h), dim=1)
#         h = F.relu(self.l0(h))
#         h = F.relu(self.l1(h))
#         return self.l2(h)

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

    # class LocalDiscriminator(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         norm = nn.BatchNorm2d
#         # norm = nn.InstanceNorm2d
#         convs = nn.Sequential(
#             nn.Conv2d(6, 64, 3, 2, 1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(64, 128, 3, 2, 1),
#             norm(128),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(128, 256, 3, 2, 1),
#             norm(256),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(256, 256, 3, 1, 1),
#             norm(256),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(256, 1, 3, 2, 1),
#         )
#         self.model = nn.Sequential(*convs)
#         init_weights(self.model, 'normal')
#
#     def forward(self, x):
#         return self.model(x)


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


# class DeepInfoAsLatent(nn.Module):
#     def __init__(self, run, epoch):
#         super().__init__()
#         model_path = Path(r'c:/data/deepinfomax/models') / Path(str(run)) / Path('encoder' + str(epoch) + '.wgt')
#         self.encoder = Encoder()
#         self.encoder.load_state_dict(torch.load(str(model_path)))
#         self.classifier = Classifier()
#
#     def forward(self, x):
#         z, features = self.encoder(x)
#         z = z.detach()
#         return self.classifier((z, features))

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

# class RGB2Depth(nn.Module):
#
#     def __init__(self, encoder='resnet18', pretrained='imagenet'):
#         super(RGB2Depth, self).__init__()
#
#         self.encoder = encoder
#         pretrained = pretrained
#         self.dim_noise = 256
#         self.avg_pool_size = 14
#
#         dims = [32, 64, 128, 256, 512, 1024, 2048]
#
#         if pretrained == 'imagenet' or pretrained == 'place':
#             pretrained = True
#         else:
#             pretrained = False
#
#         if pretrained == 'place':
#             resnet = models.__dict__[encoder](num_classes=365)
#             load_path = '/home/dudapeng/workspace/pretrained/place/' + encoder + '_places365.pth'
#             checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
#             state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
#             resnet.load_state_dict(state_dict)
#             print('place {0} loaded....'.format(encoder))
#         else:
#             resnet = models.__dict__[encoder](pretrained=pretrained)
#             print('{0} pretrained:{1}'.format(encoder, str(pretrained)))
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool  # 1/4
#         self.layer1 = resnet.layer1  # 1/4
#         self.layer2 = resnet.layer2  # 1/8
#         self.layer3 = resnet.layer3  # 1/16
#         self.layer4 = resnet.layer4  # 1/32
#         self.fc = nn.Linear(512*14*14, 128)
#
#         self.build_upsample_layers(dims)
#
#         if pretrained:
#
#             init_weights(self.deconv1, 'normal')
#             init_weights(self.deconv2, 'normal')
#             init_weights(self.deconv3, 'normal')
#             init_weights(self.deconv4, 'normal')
#             init_weights(self.latlayer1, 'normal')
#             init_weights(self.latlayer2, 'normal')
#             init_weights(self.latlayer3, 'normal')
#             init_weights(self.up_image, 'normal')
#
#         elif not pretrained:
#
#             init_weights(self, 'normal')
#
#     def build_upsample_layers(self, dims):
#
#         # norm = nn.BatchNorm2d
#         norm = nn.InstanceNorm2d
#
#         self.deconv1 = UpsampleBasicBlock(dims[4], dims[3], kernel_size=1, padding=0, norm=norm)
#         self.deconv2 = UpsampleBasicBlock(dims[3], dims[2], kernel_size=1, padding=0, norm=norm)
#         self.deconv3 = UpsampleBasicBlock(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
#         self.deconv4 = UpsampleBasicBlock(dims[1], dims[1], kernel_size=3, padding=1, norm=norm)
#         self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
#         self.up_image = nn.Sequential(
#             nn.Conv2d(64, 3, 7, 1, 3, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#
#         out = {}
#         conv1 = self.conv1(x)
#         out['0'] = self.relu(self.bn1(conv1))
#         # out['0'] = self.maxpool(self.relu(self.bn1(self.conv1(source))))
#
#         out['1'] = self.layer1(out['0'])
#         out['2'] = self.layer2(out['1'])
#         out['3'] = self.layer3(out['2'])
#         out['4'] = self.layer4(out['3'])
#
#         feature = out['4']
#         z = self.fc(feature.view(feature.shape[0], -1))
#
#         skip1 = self.latlayer1(out['3'])
#         skip2 = self.latlayer2(out['2'])
#         skip3 = self.latlayer3(out['1'])
#
#         # skip1 = out['3']
#         # skip2 = out['2']
#         # skip3 = out['1']
#
#         upconv1 = self.deconv1(out['4'])
#         upconv2 = self.deconv2(upconv1 + skip1)
#         upconv3 = self.deconv3(upconv2 + skip2)
#         upconv4 = self.deconv4(upconv3 + skip3)
#
#         x = self.up_image(upconv4)
#         # return x
#         return z, feature, x

class Upsample_Interpolate(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear',
                 reduce_dim=False):
        super(Upsample_Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        if reduce_dim:
            dim_out = int(dim_out / 2)
            self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=1, stride=1, padding=0, norm=norm)
            # self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_in, kernel_size=3, stride=1, padding=1, norm=norm)
        else:
            self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding,
                                                  norm=norm)
            # self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_out, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x, activate=True):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        x = self.conv_norm_relu1(x)
        # x = self.conv_norm_relu2(x)
        return x

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