import os
import matplotlib.pyplot as plt

import PIL
import numpy as np
import torch
import copy
import cv2
import json

from pycpd import RigidRegistration
import pydiffvg

from utils_match import rm_mk_dir
from utils_match import do_affine_transform, do_pingyi_transform
from utils_match import find_max_contour_box, flood_fill, fill_shape

pydiffvg_render = pydiffvg.RenderFunction.apply


def get_contour_normalize(img,
                          img_mask_contour_max,
                          img_mask_contour_max_bbox,
                          rsz=(224, 224),
                          is_fill_shape=False):

    min_x = max(0, img_mask_contour_max_bbox[1] - 5)
    max_x = min(
        rsz[0],
        img_mask_contour_max_bbox[1] + img_mask_contour_max_bbox[3] + 5)
    min_y = max(0, img_mask_contour_max_bbox[0] - 5)
    max_y = min(
        rsz[1],
        img_mask_contour_max_bbox[0] + img_mask_contour_max_bbox[2] + 5)

    if (is_fill_shape):
        img = fill_shape(img_mask_contour=img_mask_contour_max,
                         canvas_size=(img.shape[0], img.shape[1]))

    img_cropped_bounding_rect = img[min_x:max_x, min_y:max_y]

    if (img_cropped_bounding_rect.shape[0] == 0
            or img_cropped_bounding_rect.shape[1] == 0):
        return img_cropped_bounding_rect

    # pad image to size of rsz
    pd_row_l = int((rsz[0] - img_cropped_bounding_rect.shape[0]) / 2.0)
    pd_row_r = rsz[0] - img_cropped_bounding_rect.shape[0] - pd_row_l
    pd_col_l = int((rsz[1] - img_cropped_bounding_rect.shape[1]) / 2.0)
    pd_col_r = rsz[1] - img_cropped_bounding_rect.shape[1] - pd_col_l

    img_resized = np.pad(img_cropped_bounding_rect,
                         pad_width=[(pd_row_l, pd_row_r),
                                    (pd_col_l, pd_col_r)],
                         mode='constant')

    return img_resized


def pre_contours_normd(im_fp, gt_rsz=(224, 224)):
    img_mask_rgba = PIL.Image.open(im_fp)
    img_mask_rgba = np.array(img_mask_rgba)

    img_mask1 = img_mask_rgba[:, :, 3]  # 255
    img_mask = np.ascontiguousarray(img_mask1, dtype=np.uint8)
    # ----------------------------

    sum_img_mask = np.sum(img_mask)
    if (sum_img_mask == 0):
        print("sum=0, continue!!!!!!!!!")
        return {"sum_img_mask": 0}

    is_small_area = False
    if (sum_img_mask < 20 * 255):
        is_small_area = True

    rgb_mask = img_mask_rgba[np.where(img_mask_rgba[:, :, 3] > 0)]
    average_color_row = np.average(rgb_mask, axis=0) / 255.0
    fill_color_target = torch.FloatTensor(average_color_row)

    sub_corrd_arr = np.where(img_mask_rgba[:, :, 3] > 0)
    sub_img_m_x_mean = round(np.mean(sub_corrd_arr[0]))
    sub_img_m_y_mean = round(np.mean(sub_corrd_arr[1]))

    img_mask_contours_flood_filld = flood_fill(img_mask,
                                               canvas_size=(img_mask.shape[0],
                                                            img_mask.shape[1]))

    img_mask_contour_max_st1, img_mask_contour_max_bbox1 = find_max_contour_box(
        img_mask_contours_flood_filld)
    if ((img_mask_contour_max_st1 is None)
            or (img_mask_contour_max_bbox1 is None)):
        print("contour1-1=0, continue!!!!!!!!!")
        return {"sum_img_mask": sum_img_mask, "img_mask_contour_max_st1": None}

    # -----------------------------------------

    img_mask_contours_filld_ini = fill_shape(
        img_mask_contour=img_mask_contour_max_st1,
        canvas_size=(img_mask.shape[0], img_mask.shape[1]),
        is_flood_fill=True)

    img_mask_contours_filld_ini_area = np.sum(
        img_mask_contours_filld_ini) / 255.0
    if (img_mask_contours_filld_ini_area < 20):
        is_small_area = True
    # -----------------------------------------

    # -----------------------------------------
    img_mask_contours_normd = get_contour_normalize(
        img_mask_contours_filld_ini.copy(),
        img_mask_contour_max_st1,
        img_mask_contour_max_bbox1,
        rsz=gt_rsz,
        is_fill_shape=False)

    img_mask_contour_max_st2, img_mask_contour_max_bbox2 = find_max_contour_box(
        img_mask_contours_normd)

    if ((img_mask_contour_max_st2 is None)
            or (img_mask_contour_max_bbox2 is None)):
        print("contour2=0, continue!!!!!!!!!")
        return {
            "sum_img_mask": sum_img_mask,
            "img_mask_contour_max_st1": img_mask_contour_max_st1,
            "img_mask_contour_max_st2": None
        }

    # img_mask_contour_max = img_mask_contour_max_st2
    img_mask_contours_normd_area = np.sum(img_mask_contours_normd) / 255.0

    if (img_mask_contours_normd_area < 20):
        is_small_area = True
    # -----------------------------------------

    # -----------------------------------------
    cv2_major = cv2.__version__.split('.')[0]
    if cv2_major == '3':
        _, im_flood_contours, im_hierarchy = cv2.findContours(
            img_mask_contours_flood_filld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        im_flood_contours, im_hierarchy = cv2.findContours(
            img_mask_contours_flood_filld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    merged_flood_contours = np.concatenate(im_flood_contours)

    contour_flood_convex_hull_points = cv2.convexHull(merged_flood_contours)

    # -----------------------------------------

    return {
        "sum_img_mask": sum_img_mask,
        "img_mask": img_mask,
        "fill_color_target": fill_color_target,
        "sub_img_m_x_mean": sub_img_m_x_mean,
        "sub_img_m_y_mean": sub_img_m_y_mean,
        "img_mask_contour_max_st1": img_mask_contour_max_st1,
        "contour_flood_convex_hull_points": contour_flood_convex_hull_points,
        "img_mask_contour_max_st2": img_mask_contour_max_st2,
        "is_small_area": is_small_area,
        "img_mask_contours_normd": img_mask_contours_normd,
        "img_mask_contours_normd_area": img_mask_contours_normd_area,
        "img_mask_contours_filld_ini": img_mask_contours_filld_ini
    }


def tmask_img_replace(tar_img_seg_info, signature, gt_rsz=(224, 224)):
    test_save_dir = "./test_svg_custom/test_" + signature + "/"
    # ------------------------------------------

    tar_mask_svg_path_dir = test_save_dir + \
        "tar_" + signature + "_mask_svg_path_dir/"

    svg_path_mask_zidx_dict_fp = test_save_dir + \
        "svg_" + signature + "_path_mask_zidx_dict.json"
    with open(svg_path_mask_zidx_dict_fp, encoding="utf-8") as f:
        svg_path_mask_zidx_dict = json.load(f)

    tmp_tar_img_path_info = {}
    tar_img_seg_fp_list = []
    for sgk in tar_img_seg_info.keys():
        tar_img_seg_fp_list.append(sgk)

    if(len(tar_img_seg_fp_list) == 0):
        return tmp_tar_img_path_info

    im_pre = tar_img_seg_fp_list[0].split("/")[-2]
    tar_mask_svg_path_sub_dir = tar_mask_svg_path_dir + im_pre + "/"
    rm_mk_dir(tar_mask_svg_path_sub_dir)

    mask_replace_dir = test_save_dir + "tar_" + signature + "_mask_replace/"
    mask_replace_sub_dir = mask_replace_dir + im_pre + "/"
    rm_mk_dir(mask_replace_sub_dir)

    for sub_im_fp in tar_img_seg_fp_list:
        can_affine = False
        tmp_tar_img_path_info[sub_im_fp] = {
            "can_affine": can_affine
        }

        sub_img_m = sub_im_fp.split("/")[-1]
        sub_img_m_pre = os.path.splitext(sub_img_m)[0]

        if (os.path.isdir(sub_im_fp)):
            continue

        cur_pre_contour_info = pre_contours_normd(sub_im_fp)

        cur_sum_img_mask = cur_pre_contour_info["sum_img_mask"]
        if (cur_sum_img_mask == 0):
            continue

        cur_img_mask_contour_max_st1_ini = cur_pre_contour_info[
            "img_mask_contour_max_st1"]
        if (cur_img_mask_contour_max_st1_ini is None):
            continue

        cur_img_mask_contour_max_st2 = cur_pre_contour_info[
            "img_mask_contour_max_st2"]
        if (cur_img_mask_contour_max_st2 is None):
            continue

        # --------------------------------------
        contour_flood_convex_hull_points = cur_pre_contour_info[
            "contour_flood_convex_hull_points"]
        cur_img_mask_contour_max_st1 = contour_flood_convex_hull_points
        # --------------------------------------

        cur_img_mask = cur_pre_contour_info["img_mask"]
        fill_color_target = cur_pre_contour_info["fill_color_target"]
        cur_sub_img_m_x_mean = cur_pre_contour_info["sub_img_m_x_mean"]
        cur_sub_img_m_y_mean = cur_pre_contour_info["sub_img_m_y_mean"]

        cur_is_small_area = cur_pre_contour_info["is_small_area"]

        tmp_tar_img_path_info[sub_im_fp] = {
            "sub_im_fp": sub_im_fp,
            "cur_sum_img_mask": cur_sum_img_mask,
            "cur_is_small_area": cur_is_small_area,
        }

        # --------------------------------------
        cur_img_mask_contour_max = np.squeeze(cur_img_mask_contour_max_st1,
                                              axis=1)

        cur_img_mask_contour_max = torch.tensor(
            cur_img_mask_contour_max, dtype=torch.float32).to("cuda")

        # 转换为np为Rigid变换做准备
        np_cur_img_mask_contour_max = cur_img_mask_contour_max.detach(
        ).cpu().numpy()
        # ---------------------------------------

        len_cur_img_mask_contour_max = cur_img_mask_contour_max.shape[0]
        if (len_cur_img_mask_contour_max < 3):
            continue

        most_s_reg = None
        most_R_reg = None
        most_t_reg = None

        overl_ref_fp = tar_img_seg_info[sub_im_fp]["whole_path_img_mask_fp"]

        overl_ref_fn_pre = overl_ref_fp.split("/")[-2]
        overl_ref_cnt_fn = overl_ref_fp.split(
            "/")[-1].split(".")[0].split("_")[0]
        ref_fp = test_save_dir + "svg_" + signature + "_ref_path" + "/" + \
            overl_ref_fn_pre + "/" + overl_ref_cnt_fn + ".png"

        ref_pre_contour_info = pre_contours_normd(ref_fp)

        ref_sum_img_mask = ref_pre_contour_info["sum_img_mask"]
        if (ref_sum_img_mask == 0):
            continue

        ref_img_mask_contour_max_st1 = ref_pre_contour_info[
            "img_mask_contour_max_st1"]
        if (ref_img_mask_contour_max_st1 is None):
            continue

        ref_img_mask_contour_max_st2 = ref_pre_contour_info[
            "img_mask_contour_max_st2"]
        if (ref_img_mask_contour_max_st2 is None):
            continue

        ref_img_mask = ref_pre_contour_info["img_mask"]
        ref_sub_img_m_x_mean = ref_pre_contour_info[
            "sub_img_m_x_mean"]
        ref_sub_img_m_y_mean = ref_pre_contour_info[
            "sub_img_m_y_mean"]

        ref_img_mask_contour_max = np.squeeze(
            ref_img_mask_contour_max_st1, axis=1)

        ref_img_mask_contour_max = torch.tensor(
            ref_img_mask_contour_max, dtype=torch.float32).to("cuda")

        diff_x = (cur_sub_img_m_x_mean - ref_sub_img_m_x_mean)
        diff_y = (cur_sub_img_m_y_mean - ref_sub_img_m_y_mean)

        np_ref_img_mask_contour_max_pingyi = do_pingyi_transform(
            pts_set_src=ref_img_mask_contour_max, dx=diff_y, dy=diff_x)

        np_ref_img_mask_contour_max = np_ref_img_mask_contour_max_pingyi
        ref_img_mask_contour_max = torch.tensor(
            np_ref_img_mask_contour_max,
            dtype=torch.float32).to("cuda")

        # create a RigidRegistration object
        cur_reg = RigidRegistration(X=np_cur_img_mask_contour_max,
                                    Y=np_ref_img_mask_contour_max)

        # run the registration & collect the results
        TY, (s_reg, R_reg, t_reg) = cur_reg.register()

        np_ref_img_mask_contour_max = do_affine_transform(
            pts_set_src=ref_img_mask_contour_max,
            s_reg=s_reg,
            R_reg=R_reg,
            t_reg=t_reg)

        ref_img_mask_contour_max = torch.tensor(
            np_ref_img_mask_contour_max,
            dtype=torch.float32).to("cuda")

        # ---------------------------------------
        most_s_reg = s_reg
        most_R_reg = R_reg
        most_t_reg = t_reg

        can_affine = True

        # -------------------------------------------

        plt.clf()
        # fig, ax = plt.subplots(1,5, figsize=(10, 3), sharex=True, sharey=True)
        fig, ax = plt.subplots(1, 5, figsize=(10, 3))

        cur_img_mask_rgba_show = cur_img_mask
        ax[0].imshow(cur_img_mask_rgba_show)
        # ax[0].set_title("cur_img")
        ax[0].set_title("cur")

        contour_flood_convex_filld = fill_shape(
            img_mask_contour=contour_flood_convex_hull_points,
            canvas_size=(cur_img_mask.shape[0], cur_img_mask.shape[1]),
            is_flood_fill=True)

        ax[1].imshow(contour_flood_convex_filld)
        ax[1].set_title("ini_convex")

        ref_img_mask_rgba_show = ref_img_mask
        ax[2].imshow(ref_img_mask_rgba_show)
        # ax[2].set_title("ini_img")
        ax[2].set_title("ini_ref")

        # --------------------------------------

        np_most_img_mask_contour_max_aff = ref_img_mask_contour_max.detach(
        ).cpu().numpy()
        np_most_img_mask_contour_max_aff = np.expand_dims(
            np_most_img_mask_contour_max_aff, axis=1)

        most_sim_ref_mask_normd_aff = fill_shape(
            img_mask_contour=np_most_img_mask_contour_max_aff,
            canvas_size=(ref_img_mask.shape[0], ref_img_mask.shape[1]))

        ax[3].imshow(most_sim_ref_mask_normd_aff)
        ax[3].set_title("aff")

        # -------------------------------------------------

        tmp_svg_img_path_info = copy.deepcopy(
            svg_path_mask_zidx_dict[ref_fp])

        svg_infile = tmp_svg_img_path_info["svg_cairo_fp"]
        cur_w, cur_h, cur_shapes, cur_shape_groups = pydiffvg.svg_to_scene(
            svg_infile)

        cur_path = None
        cur_path_points = None
        cur_path_ini_color = None

        p_cnt = 0
        cur_p_cnt = 0
        for path in cur_shapes:
            if (p_cnt == tmp_svg_img_path_info["z_idx"]):
                cur_path = path
                cur_path_points = path.points
                cur_p_cnt = p_cnt

            p_cnt += 1

        c_p_cnt = 0
        color_var = []
        for group in cur_shape_groups:
            color_var.append(group.fill_color)
            if (c_p_cnt == cur_p_cnt):
                cur_path_ini_color = group.fill_color

            c_p_cnt += 1

        s_reg = most_s_reg
        R_reg = most_R_reg
        t_reg = most_t_reg

        diff_x = (cur_sub_img_m_x_mean -
                  tmp_svg_img_path_info["msk_mean_xy"][0])
        diff_y = (cur_sub_img_m_y_mean -
                  tmp_svg_img_path_info["msk_mean_xy"][1])

        cur_path_points = cur_path_points.detach().cuda()

        cur_path_points_pingyi = do_pingyi_transform(
            pts_set_src=cur_path_points, dx=diff_y, dy=diff_x)
        cur_path_points_pingyi = torch.tensor(
            cur_path_points_pingyi, dtype=torch.float32).to("cuda")

        tmp_should_pingyi = False
        if (tmp_should_pingyi):
            cur_path_points = cur_path_points_pingyi
        else:
            cur_aff_left = torch.tensor(s_reg * R_reg,
                                        device="cuda",
                                        dtype=torch.float32)

            cur_trans_matrix = torch.tensor(t_reg,
                                            device="cuda",
                                            dtype=torch.float32)

            cur_path_points = (
                torch.mm(cur_path_points_pingyi, cur_aff_left) +
                cur_trans_matrix).contiguous()

        cur_path.points = cur_path_points

        cur_shapes = []
        cur_shape_groups = []
        cur_shapes.append(cur_path)
        # fill_color_init = color_var[tmp_svg_img_path_info["z_idx"]]

        cur_path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([0]),
            fill_color=fill_color_target,
            use_even_odd_rule=False)
        cur_shape_groups.append(cur_path_group)

        canvas_height = gt_rsz[0]
        canvas_width = gt_rsz[1]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, cur_shapes, cur_shape_groups)

        cur_path_img = pydiffvg_render(
            canvas_width,  # width
            canvas_height,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args)

        # save cur_path_img
        cur_img_rgba = PIL.Image.fromarray(
            (cur_path_img.cpu().numpy() * 255).astype(np.uint8),
            "RGBA")

        cur_svg_path_fp = tmp_svg_img_path_info["svg_path_mask_fp"]
        cur_svg_path_fp_pre = os.path.splitext(
            os.path.basename(cur_svg_path_fp))[0]
        cur_svg_path_dir = os.path.dirname(cur_svg_path_fp).split(
            "/")[-1]

        cur_svg_path_fn = sub_img_m_pre + "_" + \
            cur_svg_path_dir + "_" + cur_svg_path_fp_pre + ".png"
        tmp_mask_svg_path_mask_sub_fp = tar_mask_svg_path_sub_dir + cur_svg_path_fn
        cur_img_rgba.save(tmp_mask_svg_path_mask_sub_fp)

        cur_svg_path_fn = sub_img_m_pre + "_" + \
            cur_svg_path_dir + "_" + cur_svg_path_fp_pre + ".svg"
        tmp_mask_svg_path_sub_fp = tar_mask_svg_path_sub_dir + cur_svg_path_fn
        pydiffvg.save_svg(tmp_mask_svg_path_sub_fp, canvas_width,
                          canvas_height, cur_shapes, cur_shape_groups)
        # ------------------------------------

        ax[4].imshow(
            (cur_path_img.cpu().numpy() * 255).astype(np.uint8))
        ax[4].set_title('path_trans')

        tmp_tar_img_path_info[sub_im_fp] = {
            "can_affine": can_affine,
            "tar_svg_path_mask_sub_fp": tmp_mask_svg_path_mask_sub_fp,
            "tar_svg_path_sub_fp": tmp_mask_svg_path_sub_fp,
            "fill_color_target": fill_color_target,
            "cur_path_ini_color": cur_path_ini_color
        }

    return tmp_tar_img_path_info
