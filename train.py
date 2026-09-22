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

import torch
import torch.nn.functional as F
import numpy as np
import os, random, time, csv
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def compute_frequency_weight(gt_image, alpha=1.0):
    """Compute per-pixel frequency weight from GT image using Laplacian.

    High-frequency regions (edges, textures) get higher weight so that the
    loss gradient is amplified there, indirectly driving more Gaussian
    densification in detail-rich areas.

    Args:
        gt_image: [3, H, W] tensor on CUDA
        alpha: strength of frequency weighting (0=disabled, 1=normal)

    Returns:
        weight_map: [1, H, W] tensor with values in [1.0, 1.0+alpha],
            where 1.0 means flat region (no boost) and 1.0+alpha means
            maximum frequency (full boost).
    """
    if alpha <= 0:
        return None
    gray = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32, device=gt_image.device
    ).view(1, 1, 3, 3)
    freq_map = F.conv2d(gray, laplacian_kernel, padding=1).abs().squeeze(0)  # [1, H, W]
    # Normalize to [0, 1]
    freq_max = freq_map.max().clamp(min=1e-6)
    freq_norm = freq_map / freq_max
    # Weight: 1.0 (flat) to 1.0+alpha (high frequency)
    weight_map = 1.0 + alpha * freq_norm
    return weight_map


def compute_ms_ssim(img1, img2, levels=3):
    """Compute multi-scale SSIM between two images.

    Args:
        img1, img2: [1, 3, H, W] tensors
        levels: number of scales (default 3)

    Returns:
        ms_ssim value (scalar tensor)
    """
    weights = torch.tensor([0.3, 0.35, 0.35], device=img1.device)[:levels]
    weights = weights / weights.sum()

    ssim_values = []
    current_img1 = img1
    current_img2 = img2

    for level in range(levels):
        ssim_val = fast_ssim(current_img1, current_img2)
        ssim_values.append(ssim_val)
        if level < levels - 1:
            current_img1 = F.avg_pool2d(current_img1, kernel_size=2)
            current_img2 = F.avg_pool2d(current_img2, kernel_size=2)

    ms_ssim_val = torch.zeros(1, device=img1.device)
    for level in range(levels):
        ms_ssim_val = ms_ssim_val + weights[level] * ssim_values[level]
    return ms_ssim_val


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    global _MASK_SOURCE_PATH
    _MASK_SOURCE_PATH = dataset.source_path

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    # Disable tqdm's background monitor thread to avoid a multi-thread deadlock
    # that occurs when set_postfix/refresh contends with the monitor thread,
    # especially when stdout is redirected through tee to a non-tty file.
    tqdm.monitor_interval = 0
    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Training progress",
        mininterval=1.0,
        miniters=100,
    )
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # Initialize loss CSV logger
    loss_csv_path = os.path.join(dataset.model_path, "loss_log.csv")
    loss_csv_file = open(loss_csv_path, mode='w', newline='')
    loss_csv_writer = csv.writer(loss_csv_file)
    loss_csv_writer.writerow(["iteration", "l1_loss", "ssim", "total_loss", "ema_loss", "num_gaussians"])

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Skip iteration if no gaussians are visible (avoids gradient shape mismatch)
        if not visibility_filter.any():
            continue

        # Loss computation with frequency-aware weighting and MS-SSIM
        gt_image = viewpoint_cam.original_image.cuda()

        # Compute frequency weight map from GT (amplifies loss in detail regions)
        freq_alpha = getattr(opt, 'freq_weight_alpha', 0.0)
        freq_weight = compute_frequency_weight(gt_image, alpha=freq_alpha)

        # Semantic weight map (equipment > background, person excluded). When
        # enabled it reweights the per-pixel loss; when disabled it is None and
        # behavior is identical to before.
        semantic_weight = None
        if (getattr(dataset, 'use_semantic_weight', False)
                and hasattr(viewpoint_cam, 'semantic_weight')
                and viewpoint_cam.semantic_weight is not None):
            semantic_weight = viewpoint_cam.semantic_weight.cuda()

        if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
            mask = viewpoint_cam.gt_alpha_mask.cuda()
            masked_image = image * mask
            # Frequency-aware weighted L1
            pixel_error = torch.abs(masked_image - gt_image) * mask
            # Normalization denominator. Default is the person-mask pixel count.
            # With semantic weighting we use a *weighted average*: numerator and
            # denominator are both scaled by semantic_weight, so equipment-region
            # errors get a larger share of the loss WITHOUT shifting the overall
            # loss magnitude (which would otherwise unbalance it against SSIM and
            # the regularizers).
            norm_weight = mask
            if semantic_weight is not None:
                pixel_error = pixel_error * semantic_weight
                norm_weight = mask * semantic_weight
            if freq_weight is not None:
                pixel_error = pixel_error * freq_weight
                Ll1 = pixel_error.sum() / (norm_weight.sum() * 3 * freq_weight.mean()).clamp(min=1.0)
            else:
                Ll1 = pixel_error.sum() / (norm_weight.sum() * 3).clamp(min=1.0)
            # MS-SSIM if enabled, otherwise single-scale SSIM
            use_ms_ssim = getattr(opt, 'use_ms_ssim', False)
            if use_ms_ssim:
                ssim_value = compute_ms_ssim(masked_image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = fast_ssim(masked_image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            # No person-exclusion mask path
            pixel_error = torch.abs(image - gt_image)
            if semantic_weight is not None:
                # Weighted average: divide by sum of weights, not pixel count,
                # so the loss magnitude is preserved (see masked branch above).
                pixel_error = pixel_error * semantic_weight
                weight_sum = semantic_weight.sum()
                if freq_weight is not None:
                    pixel_error = pixel_error * freq_weight
                    Ll1 = pixel_error.sum() / (weight_sum * 3 * freq_weight.mean()).clamp(min=1.0)
                else:
                    Ll1 = pixel_error.sum() / (weight_sum * 3).clamp(min=1.0)
            elif freq_weight is not None:
                pixel_error = pixel_error * freq_weight
                Ll1 = pixel_error.mean() / freq_weight.mean()
            else:
                Ll1 = l1_loss(image, gt_image)
            use_ms_ssim = getattr(opt, 'use_ms_ssim', False)
            if use_ms_ssim:
                ssim_value = compute_ms_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
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

            # Write loss to CSV every 100 iterations
            if iteration % 100 == 0:
                loss_csv_writer.writerow([
                    iteration,
                    f"{Ll1.item():.7f}",
                    f"{ssim_value.item():.7f}",
                    f"{loss.item():.7f}",
                    f"{ema_loss_for_log:.7f}",
                    gaussians._xyz.shape[0],
                ])
                loss_csv_file.flush()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack)

                    # The multiview consistent densification of fastgs
                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Progressive pruning: after densification ends, periodically remove
            # low-contribution Gaussians with gentle opacity thresholds so the
            # remaining Gaussians can compensate through continued training.
            progressive_prune_interval = getattr(opt, 'progressive_prune_interval', 0)
            progressive_prune_ratio = getattr(opt, 'progressive_prune_ratio', 0.05)
            if (progressive_prune_interval > 0
                    and iteration > opt.densify_until_iter
                    and iteration % progressive_prune_interval == 0
                    and iteration < opt.iterations - progressive_prune_interval):
                num_before = gaussians._xyz.shape[0]
                opacity_vals = gaussians.get_opacity.squeeze()
                # Remove the bottom X% by opacity
                num_to_remove = int(num_before * progressive_prune_ratio)
                if num_to_remove > 0 and num_before > num_to_remove:
                    threshold_val = torch.kthvalue(opacity_vals, num_to_remove).values
                    prune_mask = opacity_vals <= threshold_val
                    gaussians.prune_points(prune_mask)
                    num_after = gaussians._xyz.shape[0]
                    print(f"\n[ITER {iteration}] Progressive prune: "
                          f"{num_before:,} -> {num_after:,} "
                          f"(-{num_before - num_after:,}, "
                          f"thresh={threshold_val.item():.4f})")
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    # scene.save(iteration)
    loss_csv_file.close()
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    print(f"Loss log saved to: {loss_csv_path}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets
    )

    # All done
    print("\nTraining complete.")
