import numpy as np
import random
import argparse

import os
import cairosvg
import torch
import pydiffvg

import cv2
import shutil


def rm_mk_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)


# merge two ordered arrays and keep the respective order
def merge_two_arr(arr_a, arr_b):
    arr_c = []
    i = 0
    j = 0
    while (i < len(arr_a) and j < len(arr_b)):
        if (arr_a[i] < arr_b[j]):
            arr_c.append(arr_a[i])
            i += 1
        else:
            arr_c.append(arr_b[j])
            j += 1
    if (i < len(arr_a)):
        arr_c.extend(arr_a[i:])
    if (j < len(arr_b)):
        arr_c.extend(arr_b[j:])
    return arr_c


def args_str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_init_svg(svg_fp,
                  canvas_size=(224, 224),
                  trainable_stroke=False,
                  requires_grad=False,
                  experiment_dir="./svg_ref_img/",
                  scale_fac=1.33,
                  svg_cario_dir="./svg_ref_cairo/",
                  add_circle=False,
                  save_cairo_img=False,
                  save_diffvg_img=False,
                  use_cuda=False):
    shapes = []
    shape_groups = []

    infile = svg_fp
    im_fn = infile.split('/')[-1]
    im_pre, im_ext = os.path.splitext(im_fn)

    if (save_cairo_img):
        fp_cairosvg_img = experiment_dir + "test_old_cairosvg_" + im_pre + ".png"
        cairosvg.svg2png(url=infile,
                         write_to=fp_cairosvg_img,
                         output_width=int(canvas_size[0] * scale_fac),
                         output_height=(canvas_size[1] * scale_fac))

    # fp_cairosvg_svg = experiment_dir + "test_old_cairosvg_" + im_pre + ".svg"
    fp_cairosvg_svg = svg_cario_dir + im_pre + "_cairo.svg"
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

    outfile = svg_cario_dir + im_pre + ".svg"
    pydiffvg.save_svg(outfile, canvas_width, canvas_height, shapes,
                      shape_groups)

    infile = outfile
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        infile)

    assert (len(shapes) == len(shape_groups))
    # ------------------------------------------

    if (save_diffvg_img):
        diffvg_width = canvas_size[0]
        diffvg_height = canvas_size[1]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            diffvg_width, diffvg_height, shapes, shape_groups)

        render = pydiffvg.RenderFunction.apply
        img = render(
            diffvg_width,  # width
            diffvg_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args)

        # Transform to gamma space
        # new_img_fn_pydiffvg = experiment_dir + im_pre + '_init.png'
        new_img_fn_pydiffvg = experiment_dir + im_pre + '.png'
        pydiffvg.imwrite(img.cpu(), new_img_fn_pydiffvg, gamma=1.0)

    # delete cairosvg files
    # os.remove(fp_cairosvg_img)
    # os.remove(fp_cairosvg_svg)

    point_var = []
    color_var = []
    for path in shapes:
        if (use_cuda):
            path.points = path.points.to("cuda")
        path.points.requires_grad = requires_grad
        point_var.append(path.points)
    for group in shape_groups:
        if (group.fill_color is None):
            group.fill_color = torch.FloatTensor([1.0, 1.0, 1.0, 0.0])
        if (use_cuda):
            group.fill_color = group.fill_color.to("cuda")
        group.fill_color.requires_grad = requires_grad
        color_var.append(group.fill_color)

    if trainable_stroke:
        stroke_width_var = []
        stroke_color_var = []
        for path in shapes:
            if (use_cuda):
                path.stroke_width = path.stroke_width.to("cuda")
            path.stroke_width.requires_grad = requires_grad
            stroke_width_var.append(path.stroke_width)
        for group in shape_groups:
            if (use_cuda):
                group.stroke_color = group.stroke_color.to("cuda")
            group.stroke_color.requires_grad = requires_grad
            stroke_color_var.append(group.stroke_color)
        return shapes, shape_groups, point_var, color_var, stroke_width_var, stroke_color_var
    else:
        return shapes, shape_groups, point_var, color_var


def save_path_svg(ini_path,
                  fill_color=torch.FloatTensor([0.5, 0.5, 0.5, 1.0]),
                  svg_path_fp="",
                  canvas_height=224,
                  canvas_width=224):

    tp_shapes = []
    tp_shape_groups = []
    tp_shapes.append(ini_path)
    tp_fill_color = fill_color

    tp_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor([0]),
                                        fill_color=tp_fill_color,
                                        use_even_odd_rule=False)
    tp_shape_groups.append(tp_path_group)

    if (len(svg_path_fp) > 0):
        pydiffvg.save_svg(svg_path_fp, canvas_width, canvas_height, tp_shapes,
                          tp_shape_groups)

    return tp_shapes, tp_shape_groups


def path2img(ini_path, p_cnt, h, w, color_var, svg_path_fp=""):
    if (p_cnt >= len(color_var)):
        fill_color_init = torch.FloatTensor([0.5, 0.5, 0.5, 1.0])
    else:
        fill_color_init = color_var[p_cnt]

    cur_shapes, cur_shape_groups = save_path_svg(ini_path,
                                                 fill_color=fill_color_init,
                                                 svg_path_fp=svg_path_fp,
                                                 canvas_height=h,
                                                 canvas_width=w)

    canvas_width = w
    canvas_height = h
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, cur_shapes, cur_shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(
        canvas_width,  # width
        canvas_height,  # height
        2,  # num_samples_x
        2,  # num_samples_y
        0,  # seed
        None,
        *scene_args)
    # print("img.shape = ", img.shape)  # (224, 224, 4)

    return img


def judge_mask(mask,
               gt_rsz=(224, 224),
               area_th=5,
               ratio_th=100.0,
               keep_backgrd=False):
    # mask: 0,1
    if np.all(mask == 0):
        return False

    if (np.sum(mask == True) < area_th):
        return False

    msk_x_sum = np.sum(mask, axis=0)
    msk_y_sum = np.sum(mask, axis=1)
    msk_x_max = np.max(msk_x_sum)
    msk_y_max = np.max(msk_y_sum)

    if (max(msk_x_max, msk_y_max) / min(msk_x_max, msk_y_max) > ratio_th):
        return False

    if (keep_backgrd == False):
        c_lu = mask[0][0]
        c_ru = mask[0][gt_rsz[1] - 1]
        c_lb = mask[gt_rsz[0] - 1][0]
        c_rb = mask[gt_rsz[0] - 1][gt_rsz[1] - 1]
        if (c_lu or c_ru or c_lb or c_rb):
            return False

    return True


def do_affine_transform(pts_set_src, s_reg, R_reg, t_reg):
    cur_aff_left = torch.tensor(s_reg * R_reg,
                                device="cuda",
                                dtype=torch.float32)

    cur_trans_matrix = torch.tensor(t_reg, device="cuda", dtype=torch.float32)

    pts_set_aff_np = (torch.mm(pts_set_src, cur_aff_left) +
                      cur_trans_matrix).contiguous().detach().cpu().numpy()

    return pts_set_aff_np


def do_pingyi_transform(pts_set_src, dx, dy):

    cur_aff_left = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda")
    cur_trans_matrix = torch.tensor([dx, dy], device="cuda")

    pts_set_aff_torch = (torch.mm(pts_set_src, cur_aff_left) +
                         cur_trans_matrix)

    pts_set_aff_np = pts_set_aff_torch.contiguous().detach().cpu().numpy()

    return pts_set_aff_np


def find_max_contour_box(img):
    # find contours in the thresholded image

    # Check OpenCV version
    cv2_major = cv2.__version__.split('.')[0]
    if cv2_major == '3':
        # _, im_contours, im_hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        _, im_contours, im_hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_NONE)
    else:
        im_contours, im_hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if (len(im_contours) == 0):
        return None, None

    max_area = 0
    max_area_idx = 0
    max_bounding_rect = None
    for i in range(len(im_contours)):
        bounding_rect = cv2.boundingRect(im_contours[i])
        # cur_area = bounding_rect[2] * bounding_rect[3]
        cur_area = cv2.contourArea(im_contours[i])
        if (cur_area > max_area):
            max_area = cur_area
            max_area_idx = i
            max_bounding_rect = bounding_rect

    return im_contours[max_area_idx], max_bounding_rect


def flood_fill(np_mask, canvas_size=(224, 224)):
    np_filld_copy = np_mask.copy()
    zero_filld = np.zeros((canvas_size[0] + 2, canvas_size[1] + 2), np.uint8)

    cv2.floodFill(np_filld_copy, zero_filld, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(np_filld_copy)

    im_fill_out = np_mask | im_floodfill_inv

    return im_fill_out


def fill_shape(img_mask_contour, canvas_size=(224, 224), is_flood_fill=True):

    pots_np_filld = np.zeros((canvas_size[0], canvas_size[1]), dtype=np.uint8)
    cv2.fillPoly(pots_np_filld,
                 pts=[np.array(img_mask_contour, dtype=np.int32)],
                 color=(255, 255, 255))

    if (is_flood_fill):

        im_fill_out = flood_fill(pots_np_filld, canvas_size)
        pots_np_filld = im_fill_out

    return pots_np_filld
