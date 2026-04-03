
import os
import argparse
import cv2
import os
import torch
from skimage import color
from model import Network
import torchvision.utils as vutils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default='./checkpoints/latest.pth', help='folder to saved model')
parser.add_argument('--input_path', default=r'../example/gray/',
                    help='folder to input images ')
parser.add_argument('--output_path', default=r'../example/images/',
                    help='folder to output images ')
opt = parser.parse_args()

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
def load_checkpoint(opt, model):
    print(f"=> loading checkpoint '{opt.model_path}'")

    checkpoint = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    print(f"=> loaded successfully '{opt.model_path}'")

model = Network().cuda()
load_checkpoint(opt, model)

input_root = opt.input_path
output_root = opt.output_path
os.makedirs(output_root, exist_ok=True)

for file in os.listdir(input_root):
    if file.lower().endswith(('png', 'jpg', 'JPG')):
        input_path = os.path.join(input_root, file)
        output_path = os.path.join(output_root, file)

        image = cv2.imread(input_path)
        height, width = image.shape[:2]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_l, img_ab = rgb2lab(image)

        img_l, img_ab = img2tensor([img_l, img_ab], bgr2rgb=False, float32=True)
        img_l = img_l.unsqueeze(0).cuda()
        lq_rgb = tensor_lab2rgb(torch.cat([img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)], dim=1))

        output_ab = model(lq_rgb)

        output_lab = torch.cat([img_l, output_ab], dim=1)
        rain = tensor_lab2rgb(output_lab)

        rain = rain.cpu().data[0] * 255.
        rain = np.clip(rain, 0, 255)
        rain = np.ascontiguousarray(np.transpose(rain, (1, 2, 0)))
        img = rain.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path,img)
