import sys
sys.path.append(r"./")
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
from scipy.linalg import sqrtm

from scipy import linalg
import piq
from PIL import Image, ImageDraw, ImageFont
import pyiqa
import torch
import cv2
import numpy as np
from os.path import join
import warnings

import cv2
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from os.path import join

from dkm.models.model_zoo.DKMv3 import DKMv3

DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_REPROJ_THRESHOLD = 8
DEFAULT_RANSAC_METHOD = "USAC_MAGSAC"

RANSAC_ZOO = {
    "RANSAC": cv2.RANSAC,
    "USAC_FAST": cv2.USAC_FAST,
    "USAC_MAGSAC": cv2.USAC_MAGSAC,
    "USAC_PROSAC": cv2.USAC_PROSAC,
    "USAC_DEFAULT": cv2.USAC_DEFAULT,
    "USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "USAC_ACCURATE": cv2.USAC_ACCURATE,
    "USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def resize_image(image, size, interp):
    assert interp.startswith('cv2_')
    if interp.startswith('cv2_'):
        interp = getattr(cv2, 'INTER_'+interp[len('cv2_'):].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    # elif interp.startswith('pil_'):
    #     interp = getattr(PIL.Image, interp[len('pil_'):].upper())
    #     resized = PIL.Image.fromarray(image.astype(np.uint8))
    #     resized = resized.resize(size, resample=interp)
    #     resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(
            f'Unknown interpolation {interp}.')
    return resized


def fast_make_matching_figure(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)

    return out


def fast_make_matching_overlay(data, b_id):
    color0 = (data['color0'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    color1 = (data['color1'][b_id].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)  # (rH, rW, 3)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()
    inliers = data['inliers']

    rows = 2
    margin = 2
    (h0, w0), (h1, w1) = data['hw0_i'], data['hw1_i']
    h = max(h0, h1)
    H, W = margin * (rows + 1) + h * rows, margin * 3 + w0 + w1

    # canvas
    out = 255 * np.ones((H, W), np.uint8)

    wx = [margin, margin + w0, margin + w0 + margin, margin + w0 + margin + w1]
    hx = lambda row: margin * row + h * (row-1)
    out = np.stack([out] * 3, -1)

    sh = hx(row=1)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1

    sh = hx(row=2)
    out[sh: sh + h0, wx[0]: wx[1]] = color0
    out[sh: sh + h1, wx[2]: wx[3]] = color1
    mkpts0, mkpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)

    to_color = cv2.cvtColor(gray0, cv2.COLOR_GRAY2BGR)
    color = cv2.imread('00002.jpg')
    for (x0, y0), (x1, y1) in zip(mkpts0[inliers], mkpts1[inliers]):
        c = (0, 255, 0)
        cv2.line(out, (x0, y0 + sh), (x1 + margin + w0, y1 + sh), color=c, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0, y0 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + w0, y1 + sh), 3, c, -1, lineType=cv2.LINE_AA)
        to_color[y0,x0,:] = color[y1,x1,:]

    return out, to_color


def preprocess(image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
    image = image.astype(np.float32, copy=False)
    size = image.shape[:2][::-1]
    scale = np.array([1.0, 1.0])

    if resize_max:
        scale = resize_max / max(size)
        if scale < 1.0:
            size_new = tuple(int(round(x*scale)) for x in size)
            image = resize_image(image, size_new, 'cv2_area')
            scale = np.array(size) / np.array(size_new)

    if grayscale:
        assert image.ndim == 2, image.shape
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = torch.from_numpy(image / 255.0).float()

    # assure that the size is divisible by dfactor
    size_new = tuple(map(
            lambda x: int(x // dfactor * dfactor),
            image.shape[-2:]))
    image = F.resize(image, size=size_new)
    scale = np.array(size) / np.array(size_new)[::-1]
    return image, scale


def compute_geom(data,
                 ransac_method=DEFAULT_RANSAC_METHOD,
                 ransac_reproj_threshold=DEFAULT_RANSAC_REPROJ_THRESHOLD,
                 ransac_confidence=DEFAULT_RANSAC_CONFIDENCE,
                 ransac_max_iter=DEFAULT_RANSAC_MAX_ITER,
                 ) -> dict:

    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()

    if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
        return {}

    h1, w1 = data["hw0_i"]

    geo_info = {}

    F, inliers = cv2.findFundamentalMat(
        mkpts0,
        mkpts1,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if F is not None:
        geo_info["Fundamental"] = F.tolist()

    H, _ = cv2.findHomography(
        mkpts1,
        mkpts0,
        method=RANSAC_ZOO[ransac_method],
        ransacReprojThreshold=ransac_reproj_threshold,
        confidence=ransac_confidence,
        maxIters=ransac_max_iter,
    )
    if H is not None:
        geo_info["Homography"] = H.tolist()
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            mkpts0.reshape(-1, 2),
            mkpts1.reshape(-1, 2),
            F,
            imgSize=(w1, h1),
        )
        geo_info["H1"] = H1.tolist()
        geo_info["H2"] = H2.tolist()

    return geo_info


def wrap_images(img0, img1, geo_info, geom_type):
    img0 = img0[0].permute((1, 2, 0)).cpu().numpy()[..., ::-1]
    img1 = img1[0].permute((1, 2, 0)).cpu().numpy()[..., ::-1]

    h1, w1, _ = img0.shape
    h2, w2, _ = img1.shape

    rectified_image0 = img0
    rectified_image1 = None
    H = np.array(geo_info["Homography"])
    F = np.array(geo_info["Fundamental"])

    title = []
    if geom_type == "Homography":
        rectified_image1 = cv2.warpPerspective(
            img1, H, (img0.shape[1], img0.shape[0])
        )
        title = ["Image 0", "Image 1 - warped"]
    elif geom_type == "Fundamental":
        H1, H2 = np.array(geo_info["H1"]), np.array(geo_info["H2"])
        rectified_image0 = cv2.warpPerspective(img0, H1, (w1, h1))
        rectified_image1 = cv2.warpPerspective(img1, H2, (w2, h2))
        title = ["Image 0 - warped", "Image 1 - warped"]
    else:
        print("Error: Unknown geometry type")

    fig = plot_images(
        [rectified_image0.squeeze(), rectified_image1.squeeze()],
        title,
        dpi=300,
    )

    img = fig2im(fig)

    plt.close(fig)

    return img


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, size=5, pad=0.5):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        dpi:
        size:
        pad:
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    figsize = (size * n, size * 6 / 5) if size is not None else None
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)

    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])

    fig.tight_layout(pad=pad)

    return fig


def fig2im(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype="u1")
    im = buf_ndarray.reshape(h, w, 3)
    return im
#Colorful Ness

def calculate_cf(img, **kwargs):
    """Calculate Colorfulness.
    """
    (B, G, R) = cv2.split(img.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R+G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def compare_folders(folder1, folder2):
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载 DKMv3 模型
    model = DKMv3(weights=None, h=672, w=896)

    # 加载模型权重
    checkpoints_path = join('weights', 'gim_dkm_100h.ckpt')
    state_dict = torch.load(checkpoints_path, map_location='cpu')

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        if 'encoder.net.fc' in k:
            state_dict.pop(k)

    model.load_state_dict(state_dict)
    model = model.eval().to(device)

    color_values = []
    ms_values = []

    # Get the list of file names in both folders
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))

    # Find the intersection of files in both folders
    common_files = files1.intersection(files2)

    for idx, file_name in enumerate(common_files):
        if idx%2 == 1:
            img_path0 = os.path.join(folder1, list(common_files)[idx-1])
            img_path1 = os.path.join(folder1, file_name)
            img_path2 = os.path.join(folder2,list(common_files)[idx-1])  # 修复后的a
            img_path3 = os.path.join(folder2, file_name)  # 处理后的b

            # 读取图像
            image0 = read_image(img_path0)
            image1 = read_image(img_path1)
            image2 = read_image(img_path2)
            image3 = read_image(img_path3)

            # 预处理
            image0, scale0 = preprocess(image0)
            image1, scale1 = preprocess(image1)
            image2, _ = preprocess(image2)
            image3, _ = preprocess(image3)

            image0 = image0.to(device)[None]
            image1 = image1.to(device)[None]
            image2 = image2.to(device)[None]
            image3 = image3.to(device)[None]

            # 计算匹配
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense_matches, dense_certainty = model.match(image0, image1)
                sparse_matches, mconf = model.sample(dense_matches, dense_certainty, 5000)

            height0, width0 = image0.shape[-2:]
            height1, width1 = image1.shape[-2:]

            # 过滤掉低置信度的匹配点
            conf_threshold = 0.5  # 置信度阈值
            reliable_matches = sparse_matches[mconf > conf_threshold]  # 仅保留高置信度的匹配
            print(f"原始匹配点数量: {sparse_matches.shape[0]}，可靠匹配点数量: {reliable_matches.shape[0]}")

            if reliable_matches.shape[0] != 0:

                # 提取可靠的匹配点坐标
                kpts0 = reliable_matches[:, :2]
                kpts0 = torch.stack((
                    width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1)

                kpts1 = reliable_matches[:, 2:]
                kpts1 = torch.stack((
                    width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1)

                # 计算像素差异
                def get_pixel_values(image, kpts):
                    """从图像中获取匹配点的像素值"""
                    kpts = kpts.long()
                    values = image[0, :, kpts[:, 1], kpts[:, 0]]  # 取出 RGB 颜色值
                    return values.permute(1, 0)  # 调整形状 (num_kpts, 3)

                c_values = get_pixel_values(image2, kpts0)  # 从c获取像素值
                d_values = get_pixel_values(image3, kpts1)  # 从d获取像素值

                # 计算 L1 差异
                diff = torch.abs(c_values - d_values)

                avg_diff = diff.mean().item()
                ms_values.append(avg_diff)




        file_path2 = os.path.join(folder2, file_name)
        img2 = cv2.imread(file_path2)
        color = calculate_cf(img2)
        color_values.append(color)

    avg_color = np.mean(color_values)
    avg_ms = np.mean(ms_values)


    return avg_color, avg_ms


def get_inception_activations(images, model, batch_size=32):
    model.eval()
    activations = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = torch.stack(batch).cuda()
        with torch.no_grad():
            pred = model(batch)
        activations.append(pred.cpu().numpy())
    activations = np.concatenate(activations, axis=0)
    return activations

def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid

def preprocess_image(image, transform):
    image = transform(image)
    return image

def load_images_from_folder(folder, transform):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')
        img = preprocess_image(img, transform)
        images.append(img)
    return images

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def main(folder1, folder2):
    transform = transforms.Compose([
        transforms.Resize((488, 488)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = inception_v3(pretrained=True, transform_input=False).cuda()
    model.fc = nn.Identity()

    images1 = load_images_from_folder(folder1, transform)
    images2 = load_images_from_folder(folder2, transform)

    act1 = get_inception_activations(images1, model)
    act2 = get_inception_activations(images2, model)

    fid = calculate_fid(act1, act2)
    return fid

# Example usage


dataset = 'llff'# 'llff' 'nerf'

if dataset=='360':
    scenes = ['bonsai', 'counter', 'kitchen', 'room', 'treehill','flowers','garden','bicycle','stump']
    final = '/test/'
elif dataset=='llff':
    scenes = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
    final = '/test/'
Color = []
FID = []
MS = []
for scene in scenes:
#
    folder2 = './output/'+dataset +'/'+scene+'/test/ours_30000/renders/'
    folder1 = '../'+dataset+'/'+scene+final
    avg_color, avg_ms = compare_folders(folder1, folder2)


    fid_metric = pyiqa.create_metric('fid')
    avg_fid =  main(folder2, folder1)

    Color.append(avg_color)
    MS.append(avg_ms)
    FID.append(avg_fid)

    print(f"{scene} Average Color: {avg_color}")
    print(f"{scene} Average MS: {avg_ms}")
    print(f"{scene} Average FID: {avg_fid}")


print(f"Dataset Average Colorful: {np.mean(np.array(Color))}")
print(f"Dataset Average Matching Error: {np.mean(np.array(MS))}")
print(f"Dataset Average FID: {np.mean(np.array(FID))}")
