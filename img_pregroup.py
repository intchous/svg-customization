import os
import argparse
import random
import PIL
import numpy as np
import torch

import pydiffvg
from skimage import morphology

from pixel_to_svg import save_svg
from pixel_to_svg import to_svg

import json_help
from json_help import sv_json
from utils_match import load_init_svg, path2img, save_path_svg
from utils_match import rm_mk_dir, judge_mask
from utils_match import do_pingyi_transform

from utils_pregroup import get_segmentation, toposort_path


pydiffvg_render = pydiffvg.RenderFunction.apply


def imgs_dir_group(signature, pregroup_opt, gt_rsz=(224, 224)):

    test_save_dir = "./test_svg_custom/test_" + signature + "/"
    tar_img_dir = test_save_dir + "tar_" + signature + "_img/"
    tar_img_list = os.listdir(tar_img_dir)
    random.shuffle(tar_img_list)

    # ------------------------------------------
    tar_img_seg_dir = test_save_dir + "tar_" + signature + "_img_seg/"
    rm_mk_dir(tar_img_seg_dir)

    experiment_dir = test_save_dir + "tar_" + signature + "_out/"
    rm_mk_dir(experiment_dir)

    tar_img_mask_dir = test_save_dir + "tar_" + signature + "_img_mask/"
    rm_mk_dir(tar_img_mask_dir)

    tar_mask_potrace_path_dir = test_save_dir + \
        "tar_" + signature + "_mask_potrace_path/"
    rm_mk_dir(tar_mask_potrace_path_dir)

    potrace_path_zidx_dict = {}
    for fn in tar_img_list:
        infile = tar_img_dir + fn
        im_pre, im_ext = os.path.splitext(fn)
        if os.path.isdir(infile):
            continue

        print("im_fn = ", fn)

        if (fn == ".DS_Store"):
            continue

        # --------------------------------------------
        # img_pil = PIL.Image.open(infile).convert(
        #     "RGB").resize(gt_rsz, PIL.Image.Resampling.BICUBIC)

        # img_fg_mask = remove(img_pil, only_mask=True)
        # img_fg_mask = (np.array(img_fg_mask, dtype=np.int32) > 0)
        # img_fg_mask = np.ones_like(img_fg_mask)

        # --------------------------------------------

        img_pil = PIL.Image.open(infile).convert('RGBA').resize(
            gt_rsz, PIL.Image.Resampling.BICUBIC)
        img_fg_mask = np.array(img_pil)[:, :, 3]
        img_fg_mask = (np.array(img_fg_mask, dtype=np.int32) > 0)

        # 0.8, 0.9
        fg_overlap_thresh = 0.8
        # --------------------------------------------

        rag_sigma = pregroup_opt["rag_sigma"]
        rag_mode = pregroup_opt["rag_mode"]
        rag_connectivity = pregroup_opt["rag_connectivity"]

        seg_sigma = pregroup_opt["seg_sigma"]

        seg_max_dist = pregroup_opt["seg_max_dist"]
        seg_ratio = pregroup_opt["seg_ratio"]
        seg_kernel_size = pregroup_opt["seg_kernel_size"]

        mer_thresh = pregroup_opt["mer_thresh"]

        is_merge_seg = pregroup_opt["is_merge_seg"]

        img, seg = get_segmentation(
            img_dir=tar_img_dir,
            img_fn=fn,
            img_seg_dir=tar_img_seg_dir,
            gt_rsz=gt_rsz,
            seg_max_dist=seg_max_dist,
            seg_ratio=seg_ratio,
            seg_kernel_size=seg_kernel_size,
            seg_sigma=seg_sigma,
            rag_sigma=rag_sigma,
            rag_connectivity=rag_connectivity,
            rag_mode=rag_mode,
            mer_thresh=mer_thresh,
            is_padding=False,
            is_save_seg_mask=False,
            is_judge_mask=False,
            is_flood_fill=False,
            is_merge_seg=is_merge_seg,
            img_fg_mask=img_fg_mask)

        # --------------------------------
        svg = to_svg(img, seg)

        ini_svg_fp = experiment_dir + im_pre + ".svg"
        save_svg(svg, ini_svg_fp)
        # --------------------------------

        # --------------------------------------------

        img_rec_width = img.shape[1] + int(img.shape[1] / 10) * 2
        img_rec_height = img.shape[0] + int(img.shape[0] / 10) * 2

        tmp_ca_shapes, tmp_ca_shape_groups, tmp_ca_point_var, tmp_ca_color_var = load_init_svg(
            ini_svg_fp,
            canvas_size=(img_rec_width, img_rec_height),
            trainable_stroke=False,
            requires_grad=False,
            experiment_dir=experiment_dir,
            svg_cario_dir=experiment_dir,
            add_circle=False)

        # os.remove(ini_svg_fp)
        os.remove(experiment_dir + im_pre + "_cairo.svg")

        # ----------------------------------------------------------------

        selem_example = morphology.square(pregroup_opt["morph_kernel_size"])

        # -----------------------------------------------------------------

        mask_dir = tar_img_dir[:-1] + "_mask/" + im_pre + "/"
        rm_mk_dir(mask_dir)

        potrace_path_dir = tar_mask_potrace_path_dir + im_pre + "/"
        rm_mk_dir(potrace_path_dir)

        h_pad = img_rec_height  # 224 -> 268
        w_pad = img_rec_width
        ca_cnt = 0
        cur_tar_img_svg_path_info = []
        cur_tar_shapes_ini = []
        cur_tar_shapes_ini_groups = []
        act_cnt = 0

        for ini_ca_path in tmp_ca_shapes:
            cur_path_points = ini_ca_path.points.detach().cuda()
            if (cur_path_points.shape[0] == 0):
                ca_cnt += 1
                continue

            # --------------------------------------
            left = int(img.shape[1] / 10)
            right = img.shape[1] + int(img.shape[1] / 10) - 1
            top = int(img.shape[0] / 10)
            bottom = img.shape[0] + int(img.shape[0] / 10) - 1

            # --------------------------------------

            diff_x = -top
            diff_y = -left

            # --------------------------------------

            cur_path_points_pingyi = do_pingyi_transform(
                pts_set_src=cur_path_points, dx=diff_y, dy=diff_x)
            cur_path_points_pingyi = torch.tensor(
                cur_path_points_pingyi, dtype=torch.float32).to("cuda")

            ini_ca_path.points = cur_path_points_pingyi

            tp_fill_color = tmp_ca_color_var[ca_cnt]
            tp_canvas_height = gt_rsz[0]
            tp_canvas_width = gt_rsz[1]

            # --------------------------------------

            p_img_np = path2img(ini_path=ini_ca_path,
                                p_cnt=ca_cnt,
                                # h=h_pad,
                                # w=w_pad,
                                h=tp_canvas_height,
                                w=tp_canvas_width,
                                color_var=tmp_ca_color_var).cpu().numpy()

            tmp_path_mask_alpha = (p_img_np[:, :, 3] > 0)

            # ------------------------------------------------------
            area_th = pregroup_opt["area_th"]
            ratio_th = pregroup_opt["ratio_th"]

            if (judge_mask(mask=tmp_path_mask_alpha, gt_rsz=gt_rsz, area_th=area_th, ratio_th=ratio_th) == False):
                ca_cnt += 1
                # os.remove(tp_svg_path_fp)
                continue
            # ------------------------------------------------------

            # -----------------------------------------------
            fg_overlap = img_fg_mask * tmp_path_mask_alpha
            if (np.sum(fg_overlap) / np.sum(tmp_path_mask_alpha) < fg_overlap_thresh):
                ca_cnt += 1
                # os.remove(tp_svg_path_fp)
                continue
            # -----------------------------------------------

            # -----------------------------------------------

            opened_seg = morphology.opening(tmp_path_mask_alpha, selem_example)

            morph_opened_sum = np.sum(opened_seg)

            if (morph_opened_sum == 0):
                ca_cnt += 1
                continue
            # -----------------------------------------------

            tp_svg_path_fp = potrace_path_dir + im_pre + "_seg_" + str(
                ca_cnt) + ".svg"
            save_path_svg(ini_ca_path,
                          svg_path_fp=tp_svg_path_fp,
                          fill_color=tp_fill_color,
                          canvas_height=tp_canvas_height,
                          canvas_width=tp_canvas_width)

            corrd_arr = np.where(tmp_path_mask_alpha)
            msk_x_mean = round(np.mean(corrd_arr[0]))
            msk_y_mean = round(np.mean(corrd_arr[1]))

            p_img_np_pil = PIL.Image.fromarray(
                (p_img_np * 255).astype(np.uint8), "RGBA")
            tp_svg_path_png_fp = potrace_path_dir + im_pre + "_seg_" + str(
                ca_cnt) + ".png"
            p_img_np_pil.save(tp_svg_path_png_fp)

            # -----------------------------------------------

            p_img_np_pil.save(mask_dir + im_pre + "_seg_" + str(ca_cnt) +
                              ".png")

            cur_tar_img_svg_path_info.append({
                "tar_svg_path_mask_sub_fp": tp_svg_path_png_fp,
                "tar_svg_path_sub_fp": tp_svg_path_fp,
                "fill_color_target": tp_fill_color,
            })

            cur_tar_shapes_ini.append(ini_ca_path)
            cur_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
                [act_cnt]),
                fill_color=tp_fill_color,
                use_even_odd_rule=False)
            cur_tar_shapes_ini_groups.append(cur_path_group)

            ca_cnt += 1
            act_cnt += 1

        assert ca_cnt == len(tmp_ca_shapes)

        # -----------------------------------------------

        mask_dir = tar_img_dir[:-1] + "_mask/" + im_pre + "/"
        rm_mk_dir(mask_dir)
        mask_fp_list = []

        potrace_path_dir = tar_mask_potrace_path_dir + im_pre + "/"
        rm_mk_dir(potrace_path_dir)

        ca_cnt = 0
        cur_tar_color_var = []
        for c_group in cur_tar_shapes_ini_groups:
            cur_tar_color_var.append(c_group.fill_color)

        for ini_ca_path in cur_tar_shapes_ini:
            tp_fill_color = cur_tar_color_var[ca_cnt]
            tp_canvas_height = gt_rsz[0]
            tp_canvas_width = gt_rsz[1]

            tp_svg_path_fp = potrace_path_dir + im_pre + "_seg_" + str(
                ca_cnt) + ".svg"
            save_path_svg(ini_ca_path,
                          svg_path_fp=tp_svg_path_fp,
                          fill_color=tp_fill_color,
                          canvas_height=tp_canvas_height,
                          canvas_width=tp_canvas_width)

            # --------------------------------------

            p_img_np = path2img(ini_path=ini_ca_path,
                                p_cnt=ca_cnt,
                                h=tp_canvas_height,
                                w=tp_canvas_width,
                                color_var=cur_tar_color_var).cpu().numpy()

            tmp_path_mask_alpha = (p_img_np[:, :, 3] > 0)

            corrd_arr = np.where(tmp_path_mask_alpha)
            msk_x_mean = round(np.mean(corrd_arr[0]))
            msk_y_mean = round(np.mean(corrd_arr[1]))

            p_img_np_pil = PIL.Image.fromarray(
                (p_img_np * 255).astype(np.uint8), "RGBA")
            tp_svg_path_png_fp = potrace_path_dir + im_pre + "_seg_" + str(
                ca_cnt) + ".png"
            p_img_np_pil.save(tp_svg_path_png_fp)

            # -----------------------------------------------

            tmp_mask_fp = mask_dir + im_pre + "_seg_" + str(ca_cnt) + ".png"
            p_img_np_pil.save(tmp_mask_fp)
            mask_fp_list.append(tmp_mask_fp)

            potrace_path_zidx_dict[tp_svg_path_fp] = {
                "svg_path_mask_fp": tp_svg_path_png_fp,
                "potrace_svg_path_sub_fp": tp_svg_path_fp,
                "svg_cairo_fp": ini_svg_fp,
                "fill_color_target": tp_fill_color,
                "z_idx": ca_cnt,
                "msk_mean_xy": [msk_x_mean, msk_y_mean]
            }

            ca_cnt += 1

        # -----------------------------------------------

        sv_json(potrace_path_zidx_dict, test_save_dir + "svg_" + signature +
                "_potrace_path_zidx_dict.json")

        # -----------------------------------------------

        ovlp_mask_dir = tar_img_dir[:-1] + "_mask_ovlp/" + im_pre + "/"
        rm_mk_dir(ovlp_mask_dir)

        for rfp_i in range(len(mask_fp_list)):
            ref_m_fp_i = mask_fp_list[rfp_i]
            ref_m_i_fn = ref_m_fp_i.split("/")[-1]
            ref_m_i_fn_pre = ref_m_i_fn.split(".")[0]

            ref_img_mask_rgba = PIL.Image.open(ref_m_fp_i)
            ref_img_mask_rgba = np.array(ref_img_mask_rgba)
            ref_img_mask = (ref_img_mask_rgba[:, :, 3] > 0)

            for rfp_j in range(rfp_i+1, len(mask_fp_list)):
                ref_m_fp_j = mask_fp_list[rfp_j]
                mask_rgba_j = PIL.Image.open(ref_m_fp_j)
                mask_rgba_j = np.array(mask_rgba_j)
                mask_j = (mask_rgba_j[:, :, 3] > 0)

                mask_j_sum = mask_j.sum()
                if (mask_j_sum == 0):
                    continue

                tmp_img_mask = ref_img_mask * mask_j
                sum_tmp_img_mask = tmp_img_mask.sum()
                if (sum_tmp_img_mask > 0):
                    ref_img_mask = ref_img_mask * (1-tmp_img_mask)

            ref_img_mask_rgba[:, :, 3] = ref_img_mask * 255

            ref_img_mask_rgba_sv_fp = ovlp_mask_dir + ref_m_i_fn_pre + "_ovlp.png"
            ref_img_mask_rgba_sv = PIL.Image.fromarray(
                ref_img_mask_rgba.astype(np.uint8), "RGBA")
            ref_img_mask_rgba_sv.save(ref_img_mask_rgba_sv_fp)

        # -----------------------------------------------


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python img_pregroup.py --signature=animal

    parser = argparse.ArgumentParser()
    parser.add_argument("--signature", type=str, default="animal")
    args = parser.parse_args()

    pregroup_opt = json_help.parse("./img_path_match_param.yaml")
    pregroup_opt = json_help.dict_to_nonedict(pregroup_opt)

    seg_size = (224, 224)
    imgs_dir_group(signature=args.signature,
                   pregroup_opt=pregroup_opt, gt_rsz=seg_size)
