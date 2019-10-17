import os
import statistics as stats
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

import data.aligned_conc_dataset as conc_dataset
from config.default_config import DefaultConfig
from config.resnet_sunrgbd_config import RESNET_SUNRGBD_CONFIG
from data import DataProvider
from models import Contrastive_CrossModal_Conc, LocalDiscriminator, PriorDiscriminator
from collections import Counter
from torch.optim import lr_scheduler
import util.utils as util
import torchvision
import numpy as np
from collections import defaultdict


# class DeepInfoMaxLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=1.0, gamma=0.05, feat_channel=128, feat_size=28):
#         super().__init__()
#         # self.global_d = GlobalDiscriminator(256)
#         # self.local_d_inner_depth = LocalDiscriminator(feat_channel + 128)
#         # self.prior_d_rgb = PriorDiscriminator(128)
#         # self.prior_d_depth = PriorDiscriminator(128)
#
#         # self.global_d_cross = GlobalDiscriminator(512)
#         self.prior_d_cross = PriorDiscriminator(64)
#         self.local_d_inner_rgb = LocalDiscriminator(feat_channel + 64)
#         self.local_d_cross = LocalDiscriminator(6)  # + 128
#
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.feat_size = feat_size
#
#     def forward(self, data):
#
#         # z = data['z']
#         f_rgb = data['f_rgb']
#         z_rgb = data['z_rgb']
#         # f_depth = data['f_depth']
#         # z_depth = data['z_depth']
#         # fake_depth = data['fake_depth']
#
#         ms_gen = data['ms_gen']
#         # z = torch.cat((z_rgb, z_depth), 1)
#
#         # cross
#         # z_exp = z.unsqueeze(-1).unsqueeze(-1)
#         # z_exp = z_exp.expand(-1, -1, self.feat_size, self.feat_size)
#         # cross_pos = torch.cat((fake_depth, depth), 1)
#         # cross_neg = torch.cat((fake_depth, torch.cat((depth[1:], depth[0].unsqueeze(0)), dim=0)), 1)
#         # cross_pos_tmp = torch.cat((f_depth, f_rgb), 1)
#         # cross_neg_tmp = torch.cat((f_depth, torch.cat((f_rgb[1:], f_rgb[0].unsqueeze(0)), dim=0)), 1)
#         # cross_pos = torch.cat((f_rgb, f_depth), 1)
#         # cross_neg = torch.cat((f_rgb, torch.cat((f_depth[1:], f_depth[0].unsqueeze(0)), dim=0)), 1)
#         # cross_pos_tmp = torch.cat((f_depth, f_rgb), 1)
#         # cross_neg_tmp = torch.cat((f_depth, torch.cat((f_rgb[1:], f_rgb[0].unsqueeze(0)), dim=0)), 1)
#
#         z_rgb_exp = z_rgb.unsqueeze(-1).unsqueeze(-1)
#         z_rgb_exp = z_rgb_exp.expand(-1, -1, self.feat_size, self.feat_size)
#
#         # z_depth_exp = z_depth.unsqueeze(-1).unsqueeze(-1)
#         # z_depth_exp = z_depth_exp.expand(-1, -1, self.feat_size, self.feat_size)
#
#         # cross_y_M = torch.cat((cross_pos, z_rgb_exp), dim=1)
#         # cross_y_M_prime = torch.cat((cross_neg, z_rgb_exp), dim=1)
#
#         # cross_y_M_tmp = torch.cat((cross_pos_tmp, z_rgb_exp), dim=1)
#         # cross_y_M_prime_tmp = torch.cat((cross_neg_tmp, z_rgb_exp), dim=1)
#
#         # inner
#         # inner_neg_depth = torch.cat((f_depth[1:], f_depth[0].unsqueeze(0)), dim=0)
#
#
#         # inner_y_M_depth = torch.cat((f_depth, z_depth_exp), dim=1)
#         # inner_y_M_prime_depth = torch.cat((inner_neg_depth, z_depth_exp), dim=1)
#
#         prior = torch.rand_like(z_rgb)
#
#         term_a = torch.log(self.prior_d_cross(prior)).mean()
#         term_b = torch.log(1.0 - self.prior_d_cross(z_rgb)).mean()
#         PRIOR_RGB = - (term_a + term_b) * self.gamma
#         #
#         # term_a = torch.log(self.prior_d_cross(prior)).mean()
#         # term_b = torch.log(1.0 - self.prior_d_cross(z_depth)).mean()
#         # PRIOR_Depth = - (term_a + term_b) * self.gamma
#
#         # term_a = torch.log(self.prior_d_cross(prior)).mean()
#         # term_b = torch.log(1.0 - self.prior_d_cross(z)).mean()
#         # PRIOR = - (term_a + term_b) * self.gamma
#
#         inner_neg_rgb = torch.cat((f_rgb[1:], f_rgb[0].unsqueeze(0)), dim=0)
#         inner_y_M_RGB = torch.cat((f_rgb, z_rgb_exp), dim=1)
#         inner_y_M_prime_RGB = torch.cat((inner_neg_rgb, z_rgb_exp), dim=1)
#
#         Ej = -F.softplus(-self.local_d_inner_rgb(inner_y_M_RGB)).mean()
#         Em = F.softplus(self.local_d_inner_rgb(inner_y_M_prime_RGB)).mean()
#         LOCAL_RGB = (Em - Ej) * self.beta
#         #
#         # Ej = -F.softplus(-self.local_d_inner_depth(inner_y_M_depth)).mean()
#         # Em = F.softplus(self.local_d_inner_depth(inner_y_M_prime_depth)).mean()
#         # LOCAL_depth = (Em - Ej) * self.beta
#
#         LOCAL_cross = torch.zeros(1).to(device)
#         for i, (gen, _depth) in enumerate(zip(ms_gen, depth)):
#
#             if i + 1 > cfg.MULTI_SCALE_NUM:
#                 break
#
#             cross_pos = torch.cat((gen, _depth), 1)
#             cross_neg = torch.cat((gen, torch.cat((_depth[1:], _depth[0].unsqueeze(0)), dim=0)), 1)
#
#             Ej = -F.softplus(-self.local_d_cross(cross_pos)).mean()
#             Em = F.softplus(self.local_d_cross(cross_neg)).mean()
#             LOCAL_cross += (Em - Ej) * self.beta
#
#         # Ej = -F.softplus(-self.local_d_cross(cross_y_M_tmp)).mean()
#         # Em = F.softplus(self.local_d_cross(cross_y_M_prime_tmp)).mean()
#         # LOCAL_cross_tmp = (Em - Ej) * self.beta
#
#         return {'gen_loss': LOCAL_cross, 'local_rgb_loss': LOCAL_RGB, 'prior_rgb_loss': PRIOR_RGB}
#         # return LOCAL_cross + LOCAL_cross_tmp + PRIOR_RGB + PRIOR_Depth
# class DeepInfoMaxLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
#         super().__init__()
#         # self.global_d = GlobalDiscriminator(256)
#         self.local_d = LocalDiscriminator(384)
#         self.prior_d = PriorDiscriminator(128)
#
#         # self.global_d_cross = GlobalDiscriminator(512)
#         self.local_d_cross = LocalDiscriminator(640)
#
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#
#     def forward(self, y, M, M_prime, flag='inner'):
#
#         y_exp = y.unsqueeze(-1).unsqueeze(-1)
#         y_exp = y_exp.expand(-1, -1, 14, 14)
#
#         y_M = torch.cat((M, y_exp), dim=1)
#         y_M_prime = torch.cat((M_prime, y_exp), dim=1)
#
#         prior = torch.rand_like(y)
#
#         term_a = torch.log(self.prior_d(prior)).mean()
#         term_b = torch.log(1.0 - self.prior_d(y)).mean()
#         PRIOR = - (term_a + term_b) * self.gamma
#
#         if flag == 'inner':
#             Ej = -F.softplus(-self.local_d(y_M)).mean()
#             Em = F.softplus(self.local_d(y_M_prime)).mean()
#             LOCAL = (Em - Ej) * self.beta
#
#             # Ej = -F.softplus(-self.global_d(y, M)).mean()
#             # Em = F.softplus(self.global_d(y, M_prime)).mean()
#             # GLOBAL = (Em - Ej) * self.alpha
#
#         else:
#             Ej = -F.softplus(-self.local_d_cross(y_M)).mean()
#             Em = F.softplus(self.local_d_cross(y_M_prime)).mean()
#             LOCAL = (Em - Ej) * self.beta
#
#             # Ej = -F.softplus(-self.global_d_cross(y, M)).mean()
#             # Em = F.softplus(self.global_d_cross(y, M_prime)).mean()
#             # GLOBAL = (Em - Ej) * self.alpha
#
#         return LOCAL + PRIOR
#         # return LOCAL + GLOBAL + PRIOR


# class DeepInfoMaxLoss(nn.Module):
#     def __init__(self, alpha=0.5, beta=1.0, gamma=0.05):
#         super().__init__()
#         # self.global_d = GlobalDiscriminator(256)
#         self.local_d = LocalDiscriminator(512)
#         self.prior_d = PriorDiscriminator(128)
#
#         self.global_d_cross = GlobalDiscriminator(512)
#         self.local_d_cross = LocalDiscriminator(640)
#
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#
#     def forward(self, y, M, M_prime, flag='inner'):
#
#         y_exp = y.unsqueeze(-1).unsqueeze(-1)
#         y_exp = y_exp.expand(-1, -1, 14, 14)
#
#         y_M = torch.cat((M, y_exp), dim=1)
#         y_M_prime = torch.cat((M_prime, y_exp), dim=1)
#
#         prior = torch.rand_like(y)
#
#         term_a = torch.log(self.prior_d(prior)).mean()
#         term_b = torch.log(1.0 - self.prior_d(y)).mean()
#         PRIOR = - (term_a + term_b) * self.gamma
#
#         if flag == 'inner':
#             Ej = -F.softplus(-self.local_d(y_M)).mean()
#             Em = F.softplus(self.local_d(y_M_prime)).mean()
#             LOCAL = (Em - Ej) * self.beta
#
#             # Ej = -F.softplus(-self.global_d(y, M)).mean()
#             # Em = F.softplus(self.global_d(y, M_prime)).mean()
#             # GLOBAL = (Em - Ej) * self.alpha
#
#         else:
#             Ej = -F.softplus(-self.local_d_cross(y_M)).mean()
#             Em = F.softplus(self.local_d_cross(y_M_prime)).mean()
#             LOCAL = (Em - Ej) * self.beta
#
#             # Ej = -F.softplus(-self.global_d_cross(y, M)).mean()
#             # Em = F.softplus(self.global_d_cross(y, M_prime)).mean()
#             # GLOBAL = (Em - Ej) * self.alpha
#
#         return LOCAL + PRIOR
#         # return LOCAL + GLOBAL + PRIOR

# class DeepInfoMaxLoss_Cross(nn.Module):
#     def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
#         super().__init__()
#         self.global_d_cross = GlobalDiscriminator(1024)
#         self.local_d_cross = LocalDiscriminator(1152)
#
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#
#     def forward(self, y, M, M_prime):
#
#         y_exp = y.unsqueeze(-1).unsqueeze(-1)
#         y_exp = y_exp.expand(-1, -1, 7, 7)
#
#         y_M = torch.cat((M, y_exp), dim=1)
#         y_M_prime = torch.cat((M_prime, y_exp), dim=1)
#
#         Ej = -F.softplus(-self.local_d(y_M)).mean()
#         Em = F.softplus(self.local_d(y_M_prime)).mean()
#         LOCAL = (Em - Ej) * self.beta
#
#         Ej = -F.softplus(-self.global_d(y, M)).mean()
#         Em = F.softplus(self.global_d(y, M_prime)).mean()
#         GLOBAL = (Em - Ej) * self.alpha
#
#         return LOCAL + GLOBAL

    # def forward(self, y, M, M_prime):
    #
    #     # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
    #
    #     y_exp = y.unsqueeze(-1).unsqueeze(-1)
    #     y_exp = y_exp.expand(-1, -1, 26, 26)
    #
    #     y_M = torch.cat((M, y_exp), dim=1)
    #     y_M_prime = torch.cat((M_prime, y_exp), dim=1)
    #
    #     Ej = -F.softplus(-self.local_d(y_M)).mean()
    #     Em = F.softplus(self.local_d(y_M_prime)).mean()
    #     LOCAL = (Em - Ej) * self.beta
    #
    #     Ej = -F.softplus(-self.global_d(y, M)).mean()
    #     Em = F.softplus(self.global_d(y, M_prime)).mean()
    #     GLOBAL = (Em - Ej) * self.alpha
    #
    #     prior = torch.rand_like(y)
    #
    #     term_a = torch.log(self.prior_d(prior)).mean()
    #     term_b = torch.log(1.0 - self.prior_d(y)).mean()
    #     PRIOR = - (term_a + term_b) * self.gamma
    #
    #     return LOCAL + GLOBAL + PRIOR

def get_input(data, device):

    rgb = data['A'].to(device)
    depth = data['B']

    if isinstance(depth, list):
        for i, item in enumerate(depth):
            depth[i] = item.to(device)
    else:
        depth = data['B'].to(device)

    _label = data['label']
    label = torch.LongTensor(_label).to(device)
    # img_names = data['img_name']

    return rgb, depth, label

def get_scheduler(optimizer):
    print('use lambda lr')
    decay_start = cfg.NITER
    decay_iters = cfg.NITER_DECAY

    def lambda_rule(iter):
        lr_l = 1 - max(0, iter - decay_start) / float(decay_iters)
        return lr_l

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

def test_model(model, test_loader, device):

    # evaluate model on test_loader
    model.eval()
    pred_index_all_mlp_rgb = []
    pred_index_all_lin_rgb = []
    pred_index_all_mlp_depth = []
    pred_index_all_lin_depth= []
    target_index_all = []

    print('# Cls val images num = {0}'.format(len(test_loader.dataset.imgs)))

    for i, data in enumerate(test_loader):

        rgb, depth, label = get_input(data, device)
        with torch.no_grad():
            result = model(rgb, depth, class_only=True)
            lgt_glb_mlp_rgb, lgt_glb_lin_rgb = result['class_rgb']
            # lgt_glb_mlp_depth, lgt_glb_lin_depth = result['class_depth']

        pred_mlp_rgb, pred_index_mlp_rgb = util.process_output(lgt_glb_mlp_rgb.data)
        pred_lin_rgb, pred_index_lin_rgb = util.process_output(lgt_glb_lin_rgb.data)

        # pred_mlp_depth, pred_index_mlp_depth = util.process_output(lgt_glb_mlp_depth.data)
        # pred_lin_depth, pred_index_lin_depth = util.process_output(lgt_glb_lin_depth.data)

        pred_index_all_mlp_rgb.extend(list(pred_index_mlp_rgb))
        pred_index_all_lin_rgb.extend(list(pred_index_lin_rgb))
        # pred_index_all_mlp_depth.extend(list(pred_index_mlp_depth))
        # pred_index_all_lin_depth.extend(list(pred_index_lin_depth))
        target_index_all.extend(list(label.cpu().data.numpy()))

    # Mean ACC
    mean_acc_mlp_rgb = util.cal_mean_acc(cfg=cfg, data_loader=test_loader,
                                         pred_index_all=pred_index_all_mlp_rgb,
                                         target_index_all=target_index_all)
    mean_acc_lin_rgb = util.cal_mean_acc(cfg=cfg, data_loader=test_loader,
                                         pred_index_all=pred_index_all_lin_rgb,
                                         target_index_all=target_index_all)
    # mean_acc_mlp_depth = util.cal_mean_acc(cfg=cfg, data_loader=test_loader,
    #                                      pred_index_all=pred_index_all_mlp_depth,
    #                                      target_index_all=target_index_all)
    # mean_acc_lin_depth = util.cal_mean_acc(cfg=cfg, data_loader=test_loader,
    #                                      pred_index_all=pred_index_all_lin_depth,
    #                                      target_index_all=target_index_all)

    print('mean_acc_mlp_rgb: {0}, \n mean_acc_lin_rgb: {1}'.format(
          # '\n mean_acc_mlp_depth: {2}, \n mean_acc_lin_depth: {3}'.format(
        mean_acc_mlp_rgb, mean_acc_lin_rgb))
        # mean_acc_mlp_rgb, mean_acc_lin_rgb, mean_acc_mlp_depth, mean_acc_lin_depth))

    result_acc = dict()
    result_acc['mean_acc_mlp_rgb'] = mean_acc_mlp_rgb
    result_acc['mean_acc_lin_rgb'] = mean_acc_lin_rgb
    # result_acc['mean_acc_mlp_depth'] = mean_acc_mlp_depth
    # result_acc['mean_acc_lin_depth'] = mean_acc_lin_depth
    return result_acc

# def set_log_data(self, cfg):
#
#     self.loss_meters = defaultdict()
#     self.log_keys = [
#         'TRAIN_GEN_LOSS',
#         'TRAIN_LOCAL_LOSS',
#         'TRAIN_LOCAL_LOSS',
#     ]
#     for item in self.log_keys:
#         self.loss_meters[item] = AverageMeter()


if __name__ == '__main__':

    cfg = DefaultConfig()
    cfg.parse(RESNET_SUNRGBD_CONFIG().args())
    print(cfg.GPU_IDS)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_IDS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = cfg.BATCH_SIZE
    writer = SummaryWriter(log_dir=cfg.LOG_PATH)

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    # cifar_10_train_dt = CIFAR10(r'/data0/dudapeng/workspace/datasets/CIFAR10/',  download=True, transform=ToTensor())
    # cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=batch_size, shuffle=True, drop_last=True,
    #                               pin_memory=torch.cuda.is_available())

    train_transforms = list()
    train_transforms.append(conc_dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
    train_transforms.append(conc_dataset.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
    train_transforms.append(conc_dataset.RandomHorizontalFlip())
    if cfg.MULTI_SCALE:
        train_transforms.append(conc_dataset.MultiScale((cfg.FINE_SIZE, cfg.FINE_SIZE), scale_times=cfg.MULTI_SCALE_NUM))
    train_transforms.append(conc_dataset.ToTensor())
    train_transforms.append(conc_dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    train_dataset = conc_dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_TRAIN, transform=transforms.Compose(train_transforms))
    val_dataset = conc_dataset.AlignedConcDataset(cfg, data_dir=cfg.DATA_DIR_VAL, transform=transforms.Compose([
        conc_dataset.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
        conc_dataset.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
        conc_dataset.ToTensor(),
        conc_dataset.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
    train_loader = DataProvider(cfg, dataset=train_dataset)
    val_loader = DataProvider(cfg, dataset=val_dataset, shuffle=False)

    num_classes_train = list(Counter([i[1] for i in train_loader.dataset.imgs]).values())
    cfg.CLASS_WEIGHTS_TRAIN = torch.FloatTensor(num_classes_train)

    model = Contrastive_CrossModal_Conc(cfg, device=device)
    model = nn.DataParallel(model).to(device)
    optim = Adam(model.parameters(), lr=cfg.LR)
    # loss_optim = Adam(infomax_fn.parameters(), lr=2e-4)
    # cls_criterion = torch.nn.CrossEntropyLoss(cfg.CLASS_WEIGHTS_TRAIN.to(device))

    scheduler_optim = get_scheduler(optim)
    # scheduler_loss_optim = get_scheduler(loss_optim)

    epoch_restart = None
    root = None
    # epoch_restart = 860
    # root = Path(r'c:\data\deepinfomax\models\run5')

    # if epoch_restart is not None and root is not None:
    #     enc_file = root / Path('encoder' + str(epoch_restart) + '.wgt')
    #     loss_file = root / Path('loss' + str(epoch_restart) + '.wgt')
    #     encoder.load_state_dict(torch.load(str(enc_file)))
    #     loss_fn.load_state_dict(torch.load(str(loss_file)))

    iters = 0
    for epoch in range(1, 500):

        scheduler_optim.step(epoch)
        # scheduler_loss_optim.step(epoch)
        for param_group in optim.param_groups:
            lr = param_group['lr']
            print('/////////learning rate = %.7f' % lr)

        infomax_loss = []
        loss_cls = []
        loss_gen = []
        loss_local_rgb = []
        loss_prior_rgb = []
        loss_pixel = []

        model.train()
        batch = tqdm(train_loader, total=len(train_loader) // batch_size)
        # for i, data in enumerate(train_loader):
        for data in batch:
            iters += 1
            rgb, depth, label = get_input(data, device)
            # x = x.to(device)

            optim.zero_grad()
            result = model(rgb, depth, label)

            # loss_infomax = infomax_fn(result)
            gen_loss = result['gen_loss'].mean()
            local_rgb_loss = result['local_rgb_loss'].mean()
            prior_rgb_loss = result['prior_rgb_loss'].mean()
            cls_loss = result['cls_loss'].mean()

            # pixel_loss = result['pixel_loss'].mean()

            # loss = cls_loss + pixel_loss
            loss = cls_loss + gen_loss + local_rgb_loss + prior_rgb_loss

            # infomax_loss.append(gen_loss.item() + local_rgb_loss.item() + prior_rgb_loss.item())
            loss_cls.append(cls_loss.item())
            loss_gen.append(gen_loss.item())
            loss_local_rgb.append(local_rgb_loss.item())
            loss_prior_rgb.append(prior_rgb_loss.item())
            # loss_pixel.append(pixel_loss.item())

            loss.backward()
            optim.step()

            # stats.mean(infomax_loss[-cfg.BATCH_SIZE:])

            # stats.mean(cls_loss[-cfg.BATCH_SIZE:])

            # print('Train epoch ' + str(epoch) + ' Loss_infomax: {0}, Loss_cls: {1}'.format(
            #     str(stats.mean(infomax_loss[-20:])), str(stats.mean(cls_loss[-20:]))))
            batch.set_description('Train epoch ' + str(epoch) + 'Loss_cls: {0}'.format(str(stats.mean(loss_cls[-20:]))))

        writer.add_scalar('Contrastive_Loss/gen_loss', stats.mean(loss_gen), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/cls_loss', stats.mean(loss_cls), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/local_loss', stats.mean(loss_local_rgb), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/prior_loss', stats.mean(loss_prior_rgb), global_step=epoch)
        # writer.add_scalar('Contrastive_Loss/pixel_loss', stats.mean(loss_pixel), global_step=epoch)
        writer.add_image('Contrastive/rgb',
                              torchvision.utils.make_grid(rgb[:3].clone().cpu().data, 3,
                                                          normalize=True), global_step=epoch)

        if isinstance(result['ms_gen'], list):
            for i, (gen, _depth) in enumerate(zip(result['ms_gen'], depth)):
                writer.add_image('Contrastive/Gen' + str(cfg.FINE_SIZE / pow(2, i)),
                                      torchvision.utils.make_grid(gen[:6].clone().cpu().data, 3,
                                                                  normalize=True),
                                      global_step=epoch)
                writer.add_image('Contrastive/target' + str(cfg.FINE_SIZE / pow(2, i)),
                                      torchvision.utils.make_grid(_depth[:6].clone().cpu().data, 3,
                                                                  normalize=True),
                                      global_step=epoch)

        if epoch % 5 == 0:
            mean_accs = test_model(model, val_loader, device)
            mean_acc_mlp_rgb = mean_accs['mean_acc_mlp_rgb']
            mean_acc_lin_rgb = mean_accs['mean_acc_lin_rgb']
            # mean_acc_mlp_depth = mean_accs['mean_acc_mlp_depth']
            # mean_acc_lin_depth = mean_accs['mean_acc_lin_depth']
            writer.add_scalar('Contrastive_Mean_ACC/mean_acc_mlp_rgb', mean_acc_mlp_rgb, global_step=epoch)
            writer.add_scalar('Contrastive_Mean_ACC/mean_acc_lin_rgb', mean_acc_lin_rgb, global_step=epoch)
            # writer.add_scalar('Contrastive_Mean_ACC/mean_acc_mlp_depth', mean_acc_mlp_depth, global_step=epoch)
            # writer.add_scalar('Contrastive_Mean_ACC/mean_acc_lin_depth', mean_acc_lin_depth, global_step=epoch)

        # positive_show = torch.cat((rgb, gen_rgb, depth), 2)
        # negative_show = torch.cat((rgb, gen, torch.cat((depth[1:], depth[0].unsqueeze(0)), dim=0)), 2)
        # writer.add_image('contrastive_positive', torchvision.utils.make_grid(positive_show[:6].clone().cpu().data, 3, normalize=True),
        #                  global_step=totol_iters)
        # writer.add_image('contrastive_negative', torchvision.utils.make_grid(negative_show[:6].clone().cpu().data, 3, normalize=True),
        #                  global_step=totol_iters)
        # writer.add_image('contrastive_gen', torchvision.utils.make_grid(M[:6].clone().cpu().data, 3, normalize=True),
        #                  global_step=totol_iters)
        # writer.add_image('gen', torchvision.utils.make_grid(M[:6].clone().cpu().data, 3, normalize=True),
        #                  global_step=totol_iters)

        # if epoch % 10 == 0:
        #     for param_group in optim.param_groups:
        #         lr = param_group['lr']
        #         print('/////////optim learning rate = %.7f' % lr)
        #     for param_group in loss_optim.param_groups:
        #         lr = param_group['lr']
        #         print('/////////loss_optim learning rate = %.7f' % lr)
        #     root = Path(r'/home/dudapeng/workspace/checkpoints/run5')
        #     # root = Path(r'c:\data\deepinfomax\models\run5')
        #     enc_file = root / Path('encoder' + str(epoch) + '.wgt')
        #     loss_file = root / Path('loss' + str(epoch) + '.wgt')
        #     enc_file.parent.mkdir(parents=True, exist_ok=True)
        #     torch.save(encoder.state_dict(), str(enc_file))
        #     torch.save(loss_fn.state_dict(), str(loss_file))

