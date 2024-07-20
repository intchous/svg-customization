from utils import \
    get_experiment_id, \
    get_path_schedule, \
    edict_2_dict, \
    check_and_create_dir

from easydict import EasyDict as edict
import yaml

from rigid_loss import local_procrustes_loss_centered, laplacian_smoothing_loss, procrustes_distance, local_procrustes_loss_centeredv2

from multiscale_loss import gaussian_pyramid_loss
from clip_loss import Loss as ClipLoss

from utils_optm import ycrcb_conversion, linear_decay_lrlambda_f

import cairosvg
import copy
import shutil
import numpy.random as npr
import numpy as np
import os.path as osp
import os
import PIL.Image
import PIL
import pydiffvg
import torch
import cv2
import matplotlib.pyplot as plt
import random
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import warnings
warnings.filterwarnings("ignore")


pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_print_timing(False)
gamma = 1.0

##########
# helper #
##########


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--signature", type=str, default="bird")
    parser.add_argument("--losssign", type=str, default="convex")

    # parser.add_argument('--signature', nargs='+', type=str, default="laba")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--experiment", type=str, default="experiment_8x1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target",
                        type=str,
                        help="target image path",
                        default="figures/laba.png")

    parser.add_argument('--log_dir',
                        metavar='DIR',
                        default="log/test_fig_deform/")

    parser.add_argument('--initial',
                        type=str,
                        default="random",
                        choices=['random', 'circle'])

    parser.add_argument('--seginit', nargs='+', type=str)
    parser.add_argument("--num_segments", type=int, default=4)

    cfg = edict()
    args = parser.parse_args()
    cfg.debug = args.debug
    cfg.config = args.config
    cfg.experiment = args.experiment
    cfg.seed = args.seed
    cfg.target = args.target
    cfg.log_dir = args.log_dir
    cfg.initial = args.initial

    cfg.signature = args.signature
    cfg.losssign = args.losssign

    # set cfg num_segments in command
    cfg.num_segments = args.num_segments
    if args.seginit is not None:
        cfg.seginit = edict()
        cfg.seginit.type = args.seginit[0]
        if cfg.seginit.type == 'circle':
            cfg.seginit.radius = float(args.seginit[1])
    return cfg


def load_target(fp, size):
    target = PIL.Image.open(fp)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    transforms_ = []
    transforms_.append(transforms.Resize(size,
                                         interpolation=PIL.Image.BICUBIC))

    data_transforms = transforms.Compose(transforms_)

    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    gt = data_transforms(target).unsqueeze(0).to(device)
    # print("gt.shape = ", gt.shape)
    return gt


def load_init_svg(svg_fp,
                  canvas_size=(224, 224),
                  trainable_stroke=False,
                  requires_grad=False,
                  scale_fac=1.33,
                  svg_cario_dir="./svg_ref_cairo/"):

    # os.makedirs(svg_cario_dir, exist_ok=True)

    shapes = []
    shape_groups = []

    infile = svg_fp
    im_fn = infile.split('/')[-1]
    im_pre, im_ext = os.path.splitext(im_fn)

    fp_cairosvg_svg = os.path.join(svg_cario_dir, im_pre + "_cairo.svg")
    check_and_create_dir(fp_cairosvg_svg)
    cairosvg.svg2svg(url=infile,
                     write_to=fp_cairosvg_svg,
                     output_width=(canvas_size[0] * scale_fac),
                     output_height=(canvas_size[1] * scale_fac))

    # ------------------------------------------
    infile = fp_cairosvg_svg
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        infile)

    canvas_width = canvas_size[0]
    canvas_height = canvas_size[1]

    # ------------------------------------------

    outfile = os.path.join(svg_cario_dir, im_pre + ".svg")
    pydiffvg.save_svg(outfile, canvas_width, canvas_height, shapes,
                      shape_groups)

    infile = outfile
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        infile)

    assert (len(shapes) == len(shape_groups))
    # ------------------------------------------

    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        0,  # seed
        None,
        *scene_args)
    # Transform to gamma space
    pydiffvg.imwrite(img.cpu(),
                     svg_cario_dir + '/' + im_pre + '_init.png',
                     gamma=1.0)

    # delete cairosvg files
    outfile = os.path.join(svg_cario_dir, im_pre + "_init.svg")
    os.rename(infile, outfile)

    # os.remove(fp_cairosvg_img)
    os.remove(fp_cairosvg_svg)

    point_var = []
    color_var = []

    for path in shapes:
        path.points.requires_grad = requires_grad
        point_var.append(path.points)

    for group in shape_groups:
        if (group.fill_color is None):
            group.fill_color = torch.FloatTensor([1.0, 1.0, 1.0, 0.0])

        group.fill_color.requires_grad = requires_grad
        color_var.append(group.fill_color)

    if trainable_stroke:
        stroke_width_var = []
        stroke_color_var = []
        for path in shapes:
            path.stroke_width.requires_grad = requires_grad
            stroke_width_var.append(path.stroke_width)
        for group in shape_groups:
            if (group.fill_color is None):
                group.fill_color = torch.FloatTensor([1.0, 1.0, 1.0, 0.0])

            group.stroke_color.requires_grad = requires_grad
            stroke_color_var.append(group.stroke_color)
        return shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var
    else:
        return shapes, shape_groups, point_var, color_var


def main():
    ###############
    # make config #
    ###############
    global device, cfg, gt, svg_path, init_point_var
    global ini_shapes, ini_shape_groups, ini_point_var, ini_color_var
    global ini_shapes_fixed, ini_shape_groups_fixed, ini_point_var_fixed, ini_color_var_fixed

    init_point_var = []

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    # gt_rsz = (224, 224)
    gt_rsz = (cfg.optm_height, cfg.optm_width)
    gt = load_target(cfg.target, gt_rsz)
    # copy target
    new_tar_fp = osp.join(cfg.experiment_dir, im_fn)
    shutil.copyfile(cfg.target, new_tar_fp)

    if cfg.use_ycrcb:
        gt = ycrcb_conversion(gt)
    h, w = gt.shape[2:]

    path_schedule = get_path_schedule(**cfg.path_schedule)
    print("cfg.path_schedule = ", cfg.path_schedule)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        npr.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    render = pydiffvg.RenderFunction.apply

    shapes_record, shape_groups_record = [], []

    region_loss = None
    loss_matrix = []

    para_point, para_color = {}, {}
    if cfg.trainable.stroke:
        para_stroke_width, para_stroke_color = {}, {}

    pathn_record = []
    # Background
    if cfg.trainable.bg:
        # meancolor = gt.mean([2, 3])[0]
        para_bg = torch.tensor([1., 1., 1.], requires_grad=True, device=device)
    else:
        if cfg.use_ycrcb:
            para_bg = torch.tensor([219 / 255, 0, 0],
                                   requires_grad=False,
                                   device=device)
        else:
            para_bg = torch.tensor([1., 1., 1.],
                                   requires_grad=False,
                                   device=device)

    ##################
    # start_training #
    ##################

    loss_weight = None

    lrlambda_f = linear_decay_lrlambda_f(cfg.num_iter, 0.4)
    optim_schedular_dict = {}
    clip_loss_func = ClipLoss({})
    flg_stop = False
    sum_paths = 0
    best_loss = 1e6
    for path_idx, pathn in enumerate(path_schedule):

        if flg_stop:
            break

        loss_list = []
        print("=> Adding [{}] paths, [{}] ...".format(pathn, cfg.seginit.type))
        pathn_record.append(pathn)
        # pathn_record_str = '-'.join([str(i) for i in pathn_record])
        pathn_record_str = str(path_idx)

        # 使用匹配好的path初始化---------------------------------
        cur_shapes = []
        cur_shape_groups = []
        for ki in range(pathn):
            cur_shapes.append(ini_shapes[sum_paths])
            cur_shape_groups.append(ini_shape_groups[sum_paths])
            sum_paths += 1

        cur_point_var = []
        cur_color_var = []
        cur_path_var = []

        for path in cur_shapes:
            # cur_path_var也是带梯度的
            cur_path_var.append(path)
            path.points.requires_grad = True
            cur_point_var.append(path.points)

        for group in cur_shape_groups:
            group.fill_color.requires_grad = True
            cur_color_var.append(group.fill_color)

        if cfg.trainable.stroke:
            cur_stroke_width_var = []
            cur_stroke_color_var = []

            for path in cur_shapes:
                path.stroke_width.requires_grad = True
                cur_stroke_width_var.append(path.stroke_width)
            for group in cur_shape_groups:
                group.stroke_color.requires_grad = True
                cur_stroke_color_var.append(group.stroke_color)

            ini_para_stroke_width[path_idx] = cur_stroke_width_var
            ini_para_stroke_color[path_idx] = cur_stroke_color_var
        # -----------------------------------------------

        shapes_record += cur_shapes
        shape_groups_record += cur_shape_groups

        if cfg.save.init:
            filename = os.path.join(
                cfg.experiment_dir, "svg-init",
                cfg.signature + "_{}-init.svg".format(pathn_record_str))
            check_and_create_dir(filename)
            pydiffvg.save_svg(filename, w, h, shapes_record,
                              shape_groups_record)

        para = {}
        if (cfg.trainable.bg) and (path_idx == 0):
            para['bg'] = [para_bg]
        para['point'] = cur_point_var
        para['color'] = cur_color_var
        if cfg.trainable.stroke:
            para['stroke_width'] = cur_stroke_width_var
            para['stroke_color'] = cur_stroke_color_var

        pg = [{
            'params': para[ki],
            'lr': cfg.lr_base[ki]
        } for ki in sorted(para.keys())]
        optim = torch.optim.Adam(pg)

        if cfg.trainable.record:
            scheduler = LambdaLR(optim, lr_lambda=lrlambda_f, last_epoch=-1)
        else:
            scheduler = LambdaLR(optim,
                                 lr_lambda=lrlambda_f,
                                 last_epoch=cfg.num_iter)
        optim_schedular_dict[path_idx] = (optim, scheduler)

        # Inner loop training
        t_range = tqdm(range(cfg.num_iter))
        for t in t_range:

            for _, (optim, _) in optim_schedular_dict.items():
                optim.zero_grad()

            # Forward pass: render the image.
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_record, shape_groups_record)
            img = render(w, h, 2, 2, t, None, *scene_args)

            # Compose img with white background
            img = img[:, :, 3:4] * img[:, :, :3] + \
                para_bg * (1 - img[:, :, 3:4])

            if cfg.save.video:
                filename = os.path.join(
                    cfg.experiment_dir, "video-png",
                    "{}-iter{}.png".format(pathn_record_str, t))
                check_and_create_dir(filename)
                if cfg.use_ycrcb:
                    imshow = ycrcb_conversion(img,
                                              format='[2D x 3]',
                                              reverse=True).detach().cpu()
                else:
                    imshow = img.detach().cpu()
                pydiffvg.imwrite(imshow, filename, gamma=gamma)

            x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW

            if cfg.use_ycrcb:
                color_reweight = torch.FloatTensor(
                    [255 / 219, 255 / 224, 255 / 255]).to(device)
                loss = ((x - gt) * (color_reweight.view(1, -1, 1, 1)))**2
            else:
                loss = ((x - gt)**2)

            if cfg.loss.use_l1_loss:
                loss = abs(x - gt)

            if loss_weight is None:
                loss = loss.sum(1).mean()
            else:
                loss = (loss.sum(1) * loss_weight).mean()

            print("mse_loss = ", loss)

            # ------------------------------------------------

            if (best_loss > loss.item()):
                best_loss = loss.item()

                filename = os.path.join(
                    cfg.experiment_dir, "output-svg", cfg.signature + "_best.svg")
                check_and_create_dir(filename)
                pydiffvg.save_svg(filename, w, h, shapes_record,
                                  shape_groups_record)

            if (t % 10 == 0):
                filename = os.path.join(
                    cfg.experiment_dir, "output-svg", cfg.signature + "_" + str(t) + ".svg")
                check_and_create_dir(filename)
                pydiffvg.save_svg(filename, w, h, shapes_record,
                                  shape_groups_record)

            # ------------------------------------------------
            clip_loss = 0.0
            # cfg.loss.clip_loss_weight = 1.0
            cfg.loss.clip_loss_weight = 0.0
            if (cfg.loss.clip_loss_weight > 0):
                clip_losses_dict = clip_loss_func(x, gt, 0)
                clip_loss = sum(list(
                    clip_losses_dict.values())) * cfg.loss.clip_loss_weight
                # t_range.set_postfix({'clip_loss': clip_loss.item()})

            # ------------------------------------------------
            # cfg.loss.pym_loss_weight = 0.1
            cfg.loss.pym_loss_weight = 300.0
            if (cfg.loss.pym_loss_weight > 0):
                pym_loss = gaussian_pyramid_loss(
                    x, gt) * cfg.loss.pym_loss_weight

            # ------------------------------------------------

            smoothing_loss_weight = 0.0
            global_procrustes_loss_weight = 0.0

            m_procrustes_loss = 0.0
            m_smoothness_loss = 0.0
            m_global_procrustes_loss = 0.0

            if (t < cfg.num_iter / 4):
                procrustes_loss_weight = 0.01
            else:
                procrustes_loss_weight = 0.08

            if (procrustes_loss_weight > 0):
                for idx_path in range(len(cur_point_var)):
                    ini_path_pts = ini_point_var_fixed[idx_path]
                    cur_path_pts = cur_point_var[idx_path]
                    ini_path_pts = ini_path_pts.to("cuda")
                    cur_path_pts = cur_path_pts.to("cuda")

                    # -----------------------------------------------
                    cur_path_m_global_procrustes_loss = procrustes_distance(
                        ini_path_pts, cur_path_pts)

                    # 1e-4, 1e-6
                    procrustes_thresh = 1e-4
                    if (cur_path_m_global_procrustes_loss > procrustes_thresh):
                        m_global_procrustes_loss += cur_path_m_global_procrustes_loss

                    cur_path_m_procrustes_loss = local_procrustes_loss_centeredv2(
                        ini_path_pts, cur_path_pts, window_size=4, return_avg=True)

                    # 1e-4, 1e-6
                    procrustes_thresh = 1e-4
                    if (cur_path_m_procrustes_loss > procrustes_thresh):
                        m_procrustes_loss += cur_path_m_procrustes_loss

                    # -----------------------------------------------

                    cur_path_m_smoothness_loss = laplacian_smoothing_loss(
                        cur_path_pts)

                    m_smoothness_loss += cur_path_m_smoothness_loss
                    # -----------------------------------------------

                m_procrustes_loss = m_procrustes_loss * procrustes_loss_weight
                m_smoothness_loss = m_smoothness_loss * smoothing_loss_weight
                m_global_procrustes_loss = m_global_procrustes_loss * global_procrustes_loss_weight

                if (t % 10 == 0):
                    print("m_procrustes_loss:", m_procrustes_loss)
                    print("m_smoothness_loss:", m_smoothness_loss)
                    print("m_global_procrustes_loss:",
                          m_global_procrustes_loss)
                # ------------------------------------------------

            # ------------------------------------------------
            if (procrustes_loss_weight > 0 and smoothing_loss_weight > 0 and global_procrustes_loss_weight > 0):
                loss = pym_loss + clip_loss + m_procrustes_loss + \
                    m_smoothness_loss + m_global_procrustes_loss
            elif (procrustes_loss_weight > 0 and global_procrustes_loss_weight > 0):
                loss = pym_loss + clip_loss + m_procrustes_loss + m_global_procrustes_loss
            elif (procrustes_loss_weight > 0):
                loss = pym_loss + clip_loss + m_procrustes_loss
            else:
                loss = pym_loss + clip_loss
            # ------------------------------------------------

            loss_list.append(loss.item())
            t_range.set_postfix({'loss': loss.item()})
            loss.backward()

            # step
            for _, (optim, scheduler) in optim_schedular_dict.items():
                optim.step()
                scheduler.step()

            for group in shape_groups_record:
                group.fill_color.data.clamp_(0.0, 1.0)

        if cfg.loss.use_distance_weighted_loss:
            loss_weight_keep = loss_weight.detach().cpu().numpy() * 1

        if not cfg.trainable.record:
            for _, pi in pg.items():
                for ppi in pi:
                    pi.require_grad = False
            optim_schedular_dict = {}

        if cfg.save.image:
            filename = os.path.join(
                cfg.experiment_dir, "demo-png",
                cfg.signature + "_{}.png".format(pathn_record_str))
            check_and_create_dir(filename)
            if cfg.use_ycrcb:
                imshow = ycrcb_conversion(img, format='[2D x 3]',
                                          reverse=True).detach().cpu()
            else:
                imshow = img.detach().cpu()
            pydiffvg.imwrite(imshow, filename, gamma=gamma)

        if cfg.save.output:
            filename = os.path.join(
                cfg.experiment_dir, "output-svg",
                cfg.signature + "_{}.svg".format(pathn_record_str))
            check_and_create_dir(filename)
            pydiffvg.save_svg(filename, w, h, shapes_record,
                              shape_groups_record)

        loss_matrix.append(loss_list)

        if cfg.save.video:
            print("saving iteration video...")
            img_array = []
            for ii in range(0, cfg.num_iter):
                filename = os.path.join(
                    cfg.experiment_dir, "video-png",
                    "{}-iter{}.png".format(pathn_record_str, ii))
                img = cv2.imread(filename)
                # cv2.putText(
                #     img, "Path:{} \nIteration:{}".format(pathn_record_str, ii),
                #     (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                img_array.append(img)

            videoname = os.path.join(
                cfg.experiment_dir, "video-avi",
                cfg.signature + "_{}.avi".format(pathn_record_str))
            check_and_create_dir(videoname)
            out = cv2.VideoWriter(
                videoname,
                # cv2.VideoWriter_fourcc(*'mp4v'),
                cv2.VideoWriter_fourcc(*'FFV1'),
                20.0,
                (w, h))
            for iii in range(len(img_array)):
                out.write(img_array[iii])
            out.release()
            # shutil.rmtree(os.path.join(cfg.experiment_dir, "video-png"))

    print("The last loss is: {}".format(loss.item()))

    ini_filename = os.path.join(
        cfg.experiment_dir, "output-svg",
        cfg.signature + "_{}.svg".format(pathn_record_str))
    out_filename = os.path.join(
        cfg.experiment_dir, cfg.signature + "_optm.svg")
    check_and_create_dir(out_filename)
    shutil.copyfile(ini_filename, out_filename)

    if cfg.save.video:
        print("saving iteration video...")
        img_array = []
        video_dir = os.path.join(cfg.experiment_dir, "video-png")
        video_fn_list = os.listdir(video_dir)

        # 过滤掉不符合格式的文件名
        video_fn_list = [
            fn for fn in video_fn_list if fn.endswith('.png') and 'iter' in fn
        ]

        video_fn_list = sorted(
            video_fn_list,
            key=lambda x: (int(x.split('-')[
                0]), int(x.split('-')[1].split('.')[0].replace('iter', ''))))

        for ii in video_fn_list:
            filename = os.path.join(video_dir, ii)
            img = cv2.imread(filename)
            # cv2.putText(
            #     img, "Path:{} \nIteration:{}".format(pathn_record_str, ii),
            #     (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            img_array.append(img)

        videoname = os.path.join(cfg.experiment_dir, "video-avi",
                                 cfg.signature + "_whole.avi")
        check_and_create_dir(videoname)
        out = cv2.VideoWriter(
            videoname,
            # cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'FFV1'),
            20.0,
            (w, h))
        for iii in range(len(img_array)):
            out.write(img_array[iii])
        out.release()
        # shutil.rmtree(os.path.join(cfg.experiment_dir, "video-png"))


if __name__ == "__main__":
    # python svg_deform.py --signature=animal --losssign=procrustes

    cfg_arg = parse_args()
    signature = cfg_arg.signature
    loss_sign = cfg_arg.losssign
    test_save_dir = "./test_" + signature + "/"
    test_save_dir = "./test_svg_custom/test_" + signature + "/"

    img_dir = test_save_dir + "tar_" + signature + "_img/"
    img_list = os.listdir(img_dir)

    svg_dir = test_save_dir + "tar_" + signature + "_img_ini_svg_aftp/"
    svg_list = os.listdir(svg_dir)

    for svg_fn in svg_list:
        svg_path = os.path.join(svg_dir, svg_fn)
        if os.path.isdir(svg_path):
            continue

        svg_pre, svg_ext = os.path.splitext(svg_fn)
        if (svg_ext != ".svg"):
            continue

        cfg_arg = parse_args()
        with open(cfg_arg.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

        cfg_default = edict(cfg['default'])
        cfg = edict(cfg[cfg_arg.experiment])
        cfg.update(cfg_default)
        cfg.update(cfg_arg)
        cfg.exid = get_experiment_id(cfg.debug)

        cfg.signature = str(svg_pre)

        cfg.optm_height = 224
        cfg.optm_width = 224

        cfg.loss.use_distance_weighted_loss = False

        cfg.experiment_dir = osp.join(
            cfg.log_dir, '{}_{}_{}'.format(cfg.exid, cfg.signature, loss_sign))

        # initialize new shapes related stuffs.
        if cfg.trainable.stroke:
            ini_para_stroke_width, ini_para_stroke_color = {}, {}
            ini_shapes, ini_shape_groups, ini_point_var, ini_color_var, ini_stroke_width_var, ini_stroke_color_var = load_init_svg(
                svg_fp=svg_path, canvas_size=(cfg.optm_height, cfg.optm_width), trainable_stroke=True, requires_grad=False, svg_cario_dir=cfg.experiment_dir)
        else:
            ini_shapes, ini_shape_groups, ini_point_var, ini_color_var = load_init_svg(
                svg_fp=svg_path, canvas_size=(cfg.optm_height, cfg.optm_width), trainable_stroke=False, requires_grad=False, svg_cario_dir=cfg.experiment_dir)

        ini_shapes_fixed, ini_shape_groups_fixed, ini_point_var_fixed, ini_color_var_fixed = load_init_svg(
            svg_fp=svg_path, canvas_size=(cfg.optm_height, cfg.optm_width), trainable_stroke=False, requires_grad=False, svg_cario_dir=cfg.experiment_dir)

        cfg.path_schedule['max_path'] = 1
        cfg.path_schedule['schedule_each'] = len(ini_shapes)

        # 300
        cfg.num_iter = 200

        for im_fn in img_list:
            if (os.path.isdir(img_dir + im_fn)):
                continue

            im_pre, _ = os.path.splitext(im_fn)
            if (im_pre != svg_pre):
                continue

            print(im_fn)
            cfg.signature = str(im_pre)
            cfg.target = os.path.join(img_dir, im_fn)

            configfile = osp.join(cfg.experiment_dir,
                                  cfg.signature + '_config.yaml')
            check_and_create_dir(configfile)
            with open(osp.join(configfile), 'w') as f:
                yaml.dump(edict_2_dict(cfg), f)

            main()
