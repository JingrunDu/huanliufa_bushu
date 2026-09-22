import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


def sampling_cameras(my_viewpoint_stack):
    ''' Randomly sample a given number of cameras from the viewpoint stack'''

    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    
    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_value)
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY = False):
    """Compute multi-view consistency scores for Gaussians to guide densification.

    For each camera in `camlist` the function renders the scene and computes a
    photometric loss and a binary metric map of high-error pixels. It accumulates
    per-Gaussian counts of views that flagged the Gaussian and a weighted
    photometric score across views.

    Args:
        camlist (list): list of viewpoint camera objects to render from.
        gaussians: current Gaussian representation (model/state) used for rendering.
        pipe: rendering pipeline/context required by `render`.
        bg: background used for rendering.
        args: runtime config containing thresholds (e.g. `loss_thresh`).
        DENSIFY (bool): whether to compute and return the importance score
            used for densification. If False, only the pruning score is computed.

    Returns:
        importance_score (Tensor): per-Gaussian integer counts of how many views
            marked the Gaussian as high-error (floor-averaged across views).
            This output is only returned if `DENSIFY` is True.
        pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
            prioritize densification (higher means worse reconstruction consistency).
    """

    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]

        gt_image = my_viewpoint_cam.original_image.cuda()

        # Apply mask to render_image so masked regions don't produce false high-error signals
        if hasattr(my_viewpoint_cam, 'gt_alpha_mask') and my_viewpoint_cam.gt_alpha_mask is not None:
            cam_mask = my_viewpoint_cam.gt_alpha_mask.cuda()
            masked_render = render_image * cam_mask
        else:
            cam_mask = None
            masked_render = render_image

        photometric_loss = compute_photometric_loss(my_viewpoint_cam, masked_render)

        get_flag = True
        l1_loss_norm = get_loss(masked_render, gt_image)
        
        metric_map = (l1_loss_norm > args.loss_thresh).int()
        # Zero out metric_map in masked regions to prevent false prune signals
        if cam_mask is not None:
            # cam_mask is [1,H,W] or [3,H,W]; metric_map is [H,W]
            mask_2d = cam_mask[0] if cam_mask.dim() == 3 else cam_mask
            metric_map = metric_map * mask_2d.int()

        # Semantic densification bias: amplify the high-error count in equipment
        # regions so equipment Gaussians get a higher importance_score and are
        # densified more aggressively (and pruned less). This is the geometric
        # half of semantic guidance, complementing the loss weighting. The
        # presence of a semantic_weight on the camera indicates semantic mode is
        # on, so no separate flag lookup is needed here.
        if (hasattr(my_viewpoint_cam, 'semantic_weight')
                and my_viewpoint_cam.semantic_weight is not None):
            sem = my_viewpoint_cam.semantic_weight.cuda()
            sem_2d = sem[0] if sem.dim() == 3 else sem
            # Equipment pixels carry the largest weight; flag them by comparing
            # against the midpoint between background and equipment weights.
            # Use an integer factor so metric_map stays Int; the CUDA rasterizer
            # requires metric_map to be of scalar type Int (float -> runtime error).
            equip_factor = int(round(getattr(args, 'densify_equipment_factor', 2.0)))
            equip_threshold = sem_2d.max() * 0.75
            equip_region = (sem_2d >= equip_threshold).int()
            metric_map = (metric_map * (1 + (equip_factor - 1) * equip_region)).int()

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))
    
    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
