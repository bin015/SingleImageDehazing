# coding: utf-8
import os

import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import OHAZE_ROOT, HAZERD_ROOT, DEMO_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy, DM2FNet_woPhy_mod, DM2FNet_woPhy_mod_channel, DM2FNet_mod_woAS_fusion, DM2FNet_mod_woAS, DM2FNet_mod_woAS_fusion_enhenced, DM2FNet_mod_woAS_enhenced
from myModel import AODNet, TestNet_v1, FFA
from datasets import SotsDataset, OHazeDataset, HazeRDDataset, DemoDataset
from torch.utils.data import DataLoader
from metrics import ImageMetrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
log_path = './ckpt/total.txt'

# model_name = 'DM2FNet'
# model_name = 'DM2FNet_woPhy'
# model_name = 'AODNet'
# model_name = 'TestNet_v1'
# model_name = 'FFA'
# model_name = 'DM2FNet_woPhy_mod3'
# model_name = 'DM2FNet_woPhy_mod_channel'
model_name = 'DM2FNet_mod_woAS_fusion'
# model_name = 'DM2FNet_mod_woAS'
# model_name = 'DM2FNet_mod_woAS_enhenced'

# train_set_name = 'O-Haze'
# train_set_name = 'RESIDE_ITS'
train_set_name = 'RESIDE_OTS'

folder_name = model_name + '_' + train_set_name

# exp_name = 'SOTS'
# exp_name = 'O-Haze'
# exp_name = 'O-Haze'
# exp_name = 'HazeRD'
exp_name = 'Demo'

args = {
    # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
    # 'snapshot': 'iter_19000_loss_0.04261_lr_0.000014',
    # 'snapshot': 'iter_20000_loss_0.04698_lr_0.000000',

    'snapshot': 'iter_16000_loss_0.11733_lr_0.000047',
}

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    # 'HazeRD': HAZERD_ROOT,
    'Demo': DEMO_ROOT,
}

to_pil = transforms.ToPILImage()
img_metrics = ImageMetrics()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        if model_name == 'DM2FNet_woPhy':
            net = DM2FNet_woPhy().cuda()
        elif model_name == 'DM2FNet':
            net = DM2FNet().cuda()
        elif model_name == 'AODNet':
            net = AODNet().cuda()
        elif model_name == 'TestNet_v1':
            net = TestNet_v1().cuda()
        elif model_name == 'FFA':
            net = FFA().cuda()
        elif model_name == 'DM2FNet_woPhy_mod_channel':
            net = DM2FNet_woPhy_mod_channel().cuda()
        elif model_name == 'DM2FNet_mod_woAS_fusion':
            net = DM2FNet_mod_woAS_fusion().cuda()
        elif model_name == 'DM2FNet_mod_woAS_fusion_enhenced':
            net = DM2FNet_mod_woAS_fusion_enhenced().cuda()
        elif model_name == 'DM2FNet_mod_woAS_enhenced':
            net = DM2FNet_mod_woAS_enhenced().cuda()
        elif model_name == 'DM2FNet_mod_woAS':
            net = DM2FNet_mod_woAS().cuda()
        elif 'DM2FNet_woPhy_mod' in model_name:
            net = DM2FNet_woPhy_mod().cuda()
        else:
            raise NotImplementedError 

        for name, root in to_test.items():
            if 'SOTS' in name:
                dataset = SotsDataset(root)
            elif 'O-Haze' in name:
                dataset = OHazeDataset(root, 'test')
            elif 'HazeRD' in name:
                dataset = HazeRDDataset(root, 'test')
            elif 'Demo' in name:
                dataset = DemoDataset(root)
            else:
                raise NotImplementedError

            # net = nn.DataParallel(net)

            if len(args['snapshot']) > 0:
                print('load snapshot \'%s\' for testing' % args['snapshot'])
                net.load_state_dict(torch.load(os.path.join(ckpt_path, folder_name, args['snapshot'] + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            mses, psnrs, ssims, ciede2000s = [], [], [], []
            loss_record = AvgMeter()

            times = []

            for idx, data in enumerate(dataloader):
                # haze_image, _, _, _, fs = data
                haze, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, folder_name,
                                         '(%s) %s_%s' % (exp_name, name, args['snapshot'])))

                haze = haze.cuda()

                start_time = time.time()
                # res = net(haze).detach()
                if 'DM2FNet' in model_name:
                    res = sliding_forward(net, haze).detach()
                elif 'FFA' in model_name:
                    res = net(haze).detach()
                else:
                    res = net(haze).detach()

                end_time = time.time()

                elapsed_time = end_time - start_time
                times.append(elapsed_time)

                # loss = criterion(res, gts.cuda())
                # loss_record.update(loss.item(), haze.size(0))

                # for i in range(len(fs)):
                #     r = res[i].cpu().numpy()
                #     gt = gts[i].cpu().numpy()

                #     metrics = img_metrics.calculate_metrics(gt, r)

                #     mses.append(metrics['MSE'])
                #     psnrs.append(metrics['PSNR'])
                #     ssims.append(metrics['SSIM'])
                #     ciede2000s.append(metrics['CIEDE2000'])
                #     print('predicting for {} ({}/{}) [{}]: MSE {:.4f}, PSNR {:.4f}, SSIM {:.4f}, CIEDE2000 {:.4f}'
                #           .format(name, idx + 1, len(dataloader), fs[i], metrics['MSE'], metrics['PSNR'], metrics['SSIM'], metrics['CIEDE2000']))

                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, folder_name,
                                     '(%s) %s_%s' % (exp_name, name, args['snapshot']), '%s.png' % f))

            # average_time = sum(times) / len(dataloader)
            # log = f"task:[{name}] L1: {loss_record.avg:.6f}, MSE: {np.mean(mses):.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f} [{folder_name}:{args['snapshot']}] avg_time:{average_time}"
            # print(log)
            # open(log_path, 'a').write(log + '\n')


if __name__ == '__main__':
    main()
