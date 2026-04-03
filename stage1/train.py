import os
import cv2
import time
import random
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile
from model import Network
from dataset import TrainDataset
from loss import L1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset', type=str, default='../example/',
                    help='dataset folder for train')
parser.add_argument('--save-epoch-freq', type=int, default=100, help='how often to save model')
parser.add_argument('--print-freq', type=int, default=20, help='how often to print training information')
parser.add_argument('--train-visual-freq', type=int, default=10, help='how often to visualize training process')
parser.add_argument('--resume', type=str, default=None, help='continue training from this checkpoint')
parser.add_argument('--start-epoch', type=int, default=1, help='start epoch')
parser.add_argument('--output-dir', type=str, default='./checkpoints', help='model saved folder')
parser.add_argument('--log-dir', type=str, default='./log', help='save visual image')
parser.add_argument('--epochs', type=int, default=1000, help='total number of epoch')
parser.add_argument('--image-size', type=int, default=256, help='image crop size')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num-workers', type=int, default=16, help='num of workers per GPU to use')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

def tensor_lab2rgb(labs, illuminant="D65", observer="2"):
    """
    Args:
        lab    : (B, C, H, W)
    Returns:
        tuple   : (C, H, W)
    """
    illuminants = \
        {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         "D65": {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         "E": {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}
    xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])

    rgb_from_xyz = np.array([[3.240481340, -0.96925495, 0.055646640], [-1.53715152, 1.875990000, -0.20404134],
                             [-0.49853633, 0.041555930, 1.057311070]])
    B, C, H, W = labs.shape
    arrs = labs.permute((0, 2, 3, 1)).contiguous()  # (B, 3, H, W) -> (B, H, W, 3)
    L, a, b = arrs[:, :, :, 0:1], arrs[:, :, :, 1:2], arrs[:, :, :, 2:]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)
    invalid = z.data < 0
    z[invalid] = 0
    xyz = torch.cat([x, y, z], dim=3)
    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.) / 7.787
    xyz_ref_white = illuminants[illuminant][observer]
    for i in range(C):
        mask_xyz[:, :, :, i] = mask_xyz[:, :, :, i] * xyz_ref_white[i]

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(B, H, W, C)
    rgb = rgb_trans.permute((0, 3, 1, 2)).contiguous()
    mask = rgb.data > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92
    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb


def build_model(opt):
    model = Network()
    model.cuda()

    for param in model.stage.encoder.parameters():
        param.requires_grad = False

    params_to_train = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params_to_train,
                                 lr=opt.lr,
                                 betas=(0.9, 0.999), eps=1e-8)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs,
                                                           eta_min=1e-6)
    l1loss = L1Loss().cuda()
    loss = {'l1': l1loss}
    return model, optimizer, loss, scheduler


def get_trainval_loader(opt):
    train_dataset = TrainDataset(opt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers)

    return train_dataloader


def update_learning_rate(optimizer, opt):
    lrd = 0.0001 / 50
    lr = opt.lr - lrd
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('update learning rate: %f -> %f' % (opt.lr, lr))
    opt.lr = lr


def save_image(epoch, name, img_lists, opt):
    data, pred, label = img_lists
    data = data.cpu().data
    pred = pred.cpu().data
    label = label.cpu().data

    data, label, pred = data * 255, label * 255, pred * 255
    pred = np.clip(pred, 0, 255)

    h, w = pred.shape[-2:]

    gen_num = (1,1)  # (4,2)
    img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
    for i in range(gen_num[0]):
        row = i * h
        for j in range(gen_num[1]):
            idx = i * gen_num[1] + j
            tmp_list = [data[idx], pred[idx], label[idx]]
            for k in range(3):
                col = (j * 3 + k) * w
                tmp = np.transpose(tmp_list[k], (1, 2, 0))
                img[row: row + h, col: col + w] = tmp

    img_file = os.path.join(opt.log_dir, '%d_%s.jpg' % (epoch, name))
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_file, img)


def load_checkpoint(opt, model, optimizer):
    print(f"=> loading checkpoint '{opt.resume}'")

    checkpoint = torch.load(opt.resume, map_location='cpu')
    opt.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"=> loaded successfully '{opt.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(epoch, model, optimizer):
    print('==> Saving Epoch: {}'.format(epoch))
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }

    file_name = os.path.join(opt.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(opt.output_dir, 'latest.pth'))


def train(epoch, model, train_loader, optimizer, train_loss, opt):
    epoch_start_time = time.time()
    model.train()
    total_epoch_train_loss = []

    for batch_iter, data in enumerate(train_loader):
        lq, ab, = data['in_img'].cuda(), data['gt_img'].cuda()
        lq_rgb = tensor_lab2rgb(torch.cat([lq, torch.zeros_like(lq), torch.zeros_like(lq)], dim=1))

        optimizer.zero_grad()
        output_ab = model(lq_rgb)
        output_lab = torch.cat([lq, output_ab], dim=1)
        output = tensor_lab2rgb(output_lab)

        target_lab = torch.cat([lq, ab], dim=1)
        target = tensor_lab2rgb(target_lab)

        losses = train_loss['l1'](output_ab, ab)
        losses.backward()
        optimizer.step()

        total_epoch_train_loss.append(losses.cpu().data)

        if (batch_iter + 1) % opt.print_freq == 0:
            print('Epoch: {}, Epoch_iter: {}, Loss: {}'.format(epoch, batch_iter + 1, losses))
        if (batch_iter + 1) % opt.train_visual_freq == 0:
            print('Saving training image epoch: {}'.format(epoch))
            save_image(epoch, 'train', [lq_rgb, output, target], opt)
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.epochs, time.time() - epoch_start_time))

    return np.mean(total_epoch_train_loss) / opt.batch_size

def main(opt):
    model, optimizer, loss, scheduler = build_model(opt)
    scheduler.step()
    train_loader = get_trainval_loader(opt)

    total_train_loss = []

    if opt.resume:
        assert os.path.isfile(opt.resume)
        load_checkpoint(opt, model, optimizer)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        epoch_train_loss = train(epoch, model, train_loader, optimizer, loss, opt)
        total_train_loss.append(epoch_train_loss)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % epoch)
            save_checkpoint(epoch, model, optimizer)
        scheduler.step()

if __name__ == '__main__':
    opt = parser.parse_args()
    main(opt)
