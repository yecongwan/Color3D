import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from data_util import RandomCrop, RandomCropOne, RandomRotation, RandomResizedCrop, RandomHorizontallyFlip, \
    RandomVerticallyFlip
import torch
import random
import torchvision.transforms as T
import albumentations as A

from skimage import color

def rgb2lab(img_rgb):
    img_lab = color.rgb2lab(img_rgb)
    img_l = img_lab[:, :, :1]
    img_ab = img_lab[:, :, 1:]
    return img_l, img_ab

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

class TrainDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.image_size = opt.image_size
        self.dataset = os.path.join(opt.train_dataset, 'train.txt')
        self.image_path = opt.train_dataset
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.trans = A.Compose([
            A.ElasticTransform(alpha=15,
                               sigma=50,
                               alpha_affine=50, interpolation=3, border_mode=4, always_apply=False, p=0.7),
            A.RandomGridShuffle(grid=(8, 8), p=0.7),
            A.Perspective(scale=(0.05, 0.15), p=0.5, keep_size=True,interpolation=cv2.INTER_CUBIC)
        ], additional_targets={'image2': 'image'})

        self.rco = RandomCropOne(256)

        self.min_ab, self.max_ab = -128, 128
        self.interval_ab = 4
        self.ab_palette = [i for i in range(self.min_ab, self.max_ab + self.interval_ab, self.interval_ab)]

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.strip()

        in_img = cv2.imread(self.image_path + img_file)
        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
        inp_img = np.array(Image.fromarray(in_img))
        w, h,c = inp_img.shape

        transformed = self.trans(image=inp_img, image2=inp_img)

        inp_img = transformed['image']
        img_l, img_ab = rgb2lab(inp_img)
        inp_img, tar_img = img2tensor([img_l, img_ab], bgr2rgb=False, float32=True)

        ps = self.image_size

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        #inp_img, tar_img = random_distortion(inp_img, tar_img)

        sample = {'in_img': inp_img, 'gt_img': tar_img}
        return sample


def update(image):
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lightness = -random.randint(20, 70)
    hlsImg[:, :, 1] = (1.0 + lightness / float(100)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1

    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR)
    return lsImg

def random_scale(input_tensor, label_tensor, min_scale=0.8, max_scale=1.2):
    scale_factor = random.uniform(min_scale, max_scale)

    c, h, w = input_tensor.shape
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    resize_transform = T.Resize((new_h, new_w))

    input_resized = resize_transform(input_tensor)
    label_resized = resize_transform(label_tensor)

    return input_resized, label_resized


def random_distortion(input_tensor, label_tensor, max_deviation=0.2):
    shear_x = random.uniform(-max_deviation, 0)
    shear_y = random.uniform(0, max_deviation)

    c, h, w = input_tensor.shape

    affine_transform = T.RandomAffine(degrees=0, translate=None, scale=None, shear=[shear_x, shear_y])

    input_distorted = affine_transform(input_tensor)
    label_distorted = affine_transform(label_tensor)

    return input_distorted, label_distorted