#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torchvision
from random import randint
from nutils.loss_utils import l1_loss, ssim, HuberLoss, EdgeLoss, CharbonnierLoss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from nutils.general_utils import safe_state
import uuid
from skimage import color
from tqdm import tqdm
from nutils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def normalize_lll(lab_tensor):
    L = lab_tensor[0] / 100.
    return torch.stack([L, L, L], dim=0)

def normalize_lab_01(lab_tensor):
    L = lab_tensor[0] / 100. # L: [0,100] -> [0,1]
    a = (lab_tensor[1] + 128) / 255.  # a: [-128,127] -> [0,1]
    b = (lab_tensor[2] + 128) / 255. # b: [-128,127] -> [0,1]
    return torch.stack([L, a, b], dim=0)

def denormalize_lab_01(lab_tensor):
    L = lab_tensor[0] * 100
    a = lab_tensor[1] * 255 - 128
    b = lab_tensor[2] * 255 - 128
    return torch.stack([L, a, b], dim=0)


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


import torch
import numpy as np


def tensor_rgb2lab(rgb_tensor, illuminant="D65", observer="2"):

    illuminants = {
        "A": {'2': (1.098466069456375, 1, 0.3558228003436005),
              '10': (1.111420406956693, 1, 0.3519978321919493)},
        "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
                '10': (0.9672062750333777, 1, 0.8142801513128616)},
        "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
                '10': (0.9579665682254781, 1, 0.9092525159847462)},
        "D65": {'2': (0.95047, 1., 1.08883),
                '10': (0.94809667673716, 1, 1.0730513595166162)},
        "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
                '10': (0.9441713925645873, 1, 1.2064272211720228)},
        "E": {'2': (1.0, 1.0, 1.0),
              '10': (1.0, 1.0, 1.0)}
    }

    xyz_from_rgb = torch.tensor([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ], dtype=rgb_tensor.dtype, device=rgb_tensor.device)

    B, C, H, W = rgb_tensor.shape
    arrs = rgb_tensor.permute((0, 2, 3, 1)).contiguous()  # (B, 3, H, W) → (B, H, W, 3)

    mask = arrs > 0.04045
    arrs = torch.where(mask, ((arrs + 0.055) / 1.055) ** 2.4, arrs / 12.92)

    xyz = torch.mm(arrs.view(-1, 3), xyz_from_rgb.T).view(B, H, W, C)

    xyz_ref_white = illuminants[illuminant][observer]
    for i in range(C):
        xyz[:, :, :, i] /= xyz_ref_white[i]

    mask = xyz > 0.008856  # (6/29)^3
    f_xyz = torch.where(mask, torch.pow(xyz, 1 / 3), (7.787 * xyz) + (16 / 116))

    L = (116 * f_xyz[:, :, :, 1]) - 16
    a = 500 * (f_xyz[:, :, :, 0] - f_xyz[:, :, :, 1])
    b = 200 * (f_xyz[:, :, :, 1] - f_xyz[:, :, :, 2])

    lab = torch.cat([L.unsqueeze(3), a.unsqueeze(3), b.unsqueeze(3)], dim=3)
    lab_tensor = lab.permute((0, 3, 1, 2)).contiguous()  # (B, H, W, 3) → (B, 3, H, W)

    return lab_tensor


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    loss_edge = EdgeLoss()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
                                                                  render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = torch.tensor(
            np.transpose(color.rgb2lab(np.transpose(gt_image.cpu().numpy(), (1, 2, 0))), (2, 0, 1))).cuda()

        if iteration < 15000:
            gt_image = normalize_lll(gt_image)
        else:
            gt_image = normalize_lab_01(gt_image)

        if iteration % 50 == 0 and iteration < 15000:
            torchvision.utils.save_image(image, 'log/' + str(iteration) + ".jpg")
        if iteration % 50 == 0 and iteration >= 15000:
            a = denormalize_lab_01(image.clip(0, 1))  #
            b = tensor_lab2rgb(a.unsqueeze(0)).squeeze(0)
            torchvision.utils.save_image(b, 'log/' + str(iteration) + ".jpg")


        if iteration < 15000:
            Ll1 = l1_loss(image, gt_image) +  (loss_edge(
                image[:1, :, :].unsqueeze(0),
                gt_image[:1, :, :].unsqueeze(0)) +   loss_edge(
                image[1:2, :, :].unsqueeze(0),
                gt_image[1:2, :, :].unsqueeze(0)) + loss_edge(
                image[2:, :, :].unsqueeze(0),
                gt_image[2:, :, :].unsqueeze(0)))/3.
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            Ll1 = l1_loss(image, gt_image)  +  loss_edge(
                image[:1, :, :].unsqueeze(0),
                gt_image[:1, :, :].unsqueeze(0))

            loss = ((1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < 20000:#opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold*1.2, 0.005, scene.cameras_extent, size_threshold)
                # opt.densify_grad_threshold*0.4
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
