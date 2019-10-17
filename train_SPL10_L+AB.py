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

import data.spl10_dataset as SPL10
from data.spl10_dataset import RGB2Lab
from config.default_config import DefaultConfig
from config.resnet_spl10_config import RESNET_SPL10_CONFIG
from data import DataProvider
from models import Contrastive_CrossModal_Conc, LocalDiscriminator, PriorDiscriminator
from collections import Counter
from torch.optim import lr_scheduler
import util.utils as util
import torchvision
import numpy as np
from collections import defaultdict

# def get_input(data, device):

#     rgb = data['A'].to(device)
#     depth = data['B']

#     if isinstance(depth, list):
#         for i, item in enumerate(depth):
#             depth[i] = item.to(device)
#     else:
#         depth = data['B'].to(device)

#     _label = data['label']
#     label = torch.LongTensor(_label).to(device)
#     # img_names = data['img_name']

#     return rgb, depth, label

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

        # rgb, depth, label = get_input(data, device)
        rgb = data['image']

        label =data['label']

        with torch.no_grad():
            result = model(rgb,class_only=True)
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
    cfg.parse(RESNET_SPL10_CONFIG().args())
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
    # normalize = Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
    #                                  std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])

    train_transforms = list()
    train_transforms.append(SPL10.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)))
    train_transforms.append(SPL10.RandomCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)))
    train_transforms.append(SPL10.RandomHorizontalFlip())
    if cfg.MULTI_SCALE:
        train_transforms.append(SPL10.MultiScale((cfg.FINE_SIZE, cfg.FINE_SIZE), scale_times=cfg.MULTI_SCALE_NUM))

    train_transforms.append(RGB2Lab()),
    train_transforms.append(SPL10.ToTensor())
    # train_transforms.append(SPL10.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    train_transforms.append(SPL10.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                     std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]))


    # train_transforms.append(normalize)


    train_dataset = SPL10.SPL10_Dataset(cfg, data_dir=cfg.DATA_DIR_TRAIN, transform=transforms.Compose(train_transforms))
    

    val_dataset = SPL10.SPL10_Dataset(cfg, data_dir=cfg.DATA_DIR_VAL, transform=transforms.Compose([
        SPL10.Resize((cfg.LOAD_SIZE, cfg.LOAD_SIZE)),
        SPL10.CenterCrop((cfg.FINE_SIZE, cfg.FINE_SIZE)),
        RGB2Lab(),
        SPL10.ToTensor(),
    	SPL10.Normalize(mean=[(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2],
                                     std=[(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2])
    ]))
    train_loader = DataProvider(cfg, dataset=train_dataset)
    val_loader = DataProvider(cfg, dataset=val_dataset, shuffle=False)

    num_classes_train = list(Counter([i[1] for i in train_loader.dataset.imgs]).values())
    cfg.CLASS_WEIGHTS_TRAIN = torch.FloatTensor(num_classes_train)

    model = Contrastive_CrossModal_Conc(cfg, device=device)
    model = nn.DataParallel(model).to(device)
    optim = Adam(model.parameters(), lr=cfg.LR)
    load_model=False
    if load_model:
        model=torch.load('./checkpoint/model_1_LAB.mdl')
        print("load pretrained model")
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
        loss_l = []
        loss_ab = []
        loss_local_rgb = []
        loss_prior_rgb = []
        loss_pixel = []

        model.train()
        batch = tqdm(train_loader, total=len(train_loader) // batch_size)
        print('# Train images num = {0}'.format(len(train_loader.dataset.imgs)))

        # for i, data in enumerate(train_loader):
        for data in batch:
            iters += 1
            # rgb, depth, label = get_input(data, device)
            image = data['image']
            Lab = data['lab']

            label =data['label']
            l = []
            ab = []
            for i in range(cfg.MULTI_SCALE_NUM):
            	_l ,_ab= torch.split(Lab[i],[1,2],dim=1)
            	l.append(_l)
            	ab.append(_ab)

            # x = x.to(device)

            optim.zero_grad()
            result = model(image,l,ab,label)

            # loss_infomax = infomax_fn(result)
            l_loss = result['gen_l_loss'].mean()
            ab_loss = result['gen_ab_loss'].mean()
            local_rgb_loss = result['local_rgb_loss'].mean()
            prior_rgb_loss = result['prior_rgb_loss'].mean()
            cls_loss = result['cls_loss'].mean()

            # pixel_loss = result['pixel_loss'].mean()

            # loss = cls_loss + pixel_loss
            # loss = cls_loss + l_loss + ab_loss + local_rgb_loss + prior_rgb_loss
            loss = l_loss + ab_loss + local_rgb_loss + prior_rgb_loss
            

            # infomax_loss.append(gen_loss.item() + local_rgb_loss.item() + prior_rgb_loss.item())
            loss_cls.append(cls_loss.item())
            loss_l.append(l_loss.item())
            loss_ab.append(ab_loss.item())
            loss_local_rgb.append(local_rgb_loss.item())
            loss_prior_rgb.append(prior_rgb_loss.item())
            # loss_pixel.append(pixel_loss.item())

            loss.backward()
            optim.step()
            # if isinstance(result['ms_gen'], list):
            #     for i, (gen, _ab) in enumerate(zip(result['ms_gen'], ab)):
            #         b,c,h,w = gen.size()
            #         pic = np.zeros((6,3,h,w))
            #         pic[:,:2,:,:] = gen[:6].clone().cpu().data
            #         # pic[:,2:,:,:] += 255
            #         writer.add_image('Contrastive/Gen' + str(cfg.FINE_SIZE / pow(2, i)),
            #                               torchvision.utils.make_grid(torch.from_numpy(pic), 3,normalize=True),
            #                               global_step=epoch)
            #         pic[:,:2,:,:] =_ab[:6].clone().cpu().data
            #         writer.add_image('Contrastive/target' + str(cfg.FINE_SIZE / pow(2, i)),
            #                               torchvision.utils.make_grid(torch.from_numpy(pic), 3,normalize=True),
            #                               global_step=epoch)



            # stats.mean(infomax_loss[-cfg.BATCH_SIZE:])

            # stats.mean(cls_loss[-cfg.BATCH_SIZE:])

            # print('Train epoch ' + str(epoch) + ' Loss_infomax: {0}, Loss_cls: {1}'.format(
            #     str(stats.mean(infomax_loss[-20:])), str(stats.mean(cls_loss[-20:]))))
            batch.set_description('Train epoch ' + str(epoch) + 'Loss_cls: {0}'.format(str(stats.mean(loss_cls[-20:]))))

        writer.add_scalar('Contrastive_Loss/l_loss', stats.mean(loss_l), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/ab_loss', stats.mean(loss_ab), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/cls_loss', stats.mean(loss_cls), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/local_loss', stats.mean(loss_local_rgb), global_step=epoch)
        writer.add_scalar('Contrastive_Loss/prior_loss', stats.mean(loss_prior_rgb), global_step=epoch)
        # writer.add_scalar('Contrastive_Loss/pixel_loss', stats.mean(loss_pixel), global_step=epoch)
        # writer.add_image('Contrastive/l',
        #                       torchvision.utils.make_grid(l[:3].clone().cpu().data, 3,
        #                                                 normalize=True), global_step=epoch)
        if isinstance(result['l_gen'], list):
            for i, (gen_l,gen_ab, l_label,ab_label) in enumerate(zip(result['l_gen'],result['ab_gen'],l,ab)):
                b,c,h,w = gen_l.size()
                pic = np.zeros((1,3,h,w))
                # pic[:,2:,:,:] += 255
                writer.add_image('Contrastive/Gen_l' + str(cfg.FINE_SIZE / pow(2, i)),
                                      torchvision.utils.make_grid(gen_l[:1].clone().cpu().data, 1,normalize=True),
                                      global_step=epoch)
                pic[:,1:,:,:] = gen_ab[:1].clone().cpu().data
                writer.add_image('Contrastive/Gen_ab' + str(cfg.FINE_SIZE / pow(2, i)),
                                      torchvision.utils.make_grid(torch.from_numpy(pic), 3,normalize=True),
                                      global_step=epoch)
                writer.add_image('Contrastive/target_l' + str(cfg.FINE_SIZE / pow(2, i)),
                                      torchvision.utils.make_grid(l_label[:1].clone().cpu().data, 1,normalize=True),
                                      global_step=epoch)
                pic[:,1:,:,:] = ab_label[:1].clone().cpu().data
                writer.add_image('Contrastive/target_ab' + str(cfg.FINE_SIZE / pow(2, i)),
                                      torchvision.utils.make_grid(torch.from_numpy(pic), 3,normalize=True),
                                      global_step=epoch)
        if epoch % 50==0:
            torch.save(model,'./checkpoint/'+'model_lab_'+str(epoch)+'.mdl')
        if epoch % 10 == 0:
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

