import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
import argparse
import shutil

from json_help import sv_json
from utils_match import rm_mk_dir, load_init_svg, path2img, find_max_contour_box, save_path_svg


def svg_mask_path_match(signature, gt_rsz=(224, 224)):

    test_save_dir = "./test_svg_custom/test_" + signature + "/"
    svg_ref_dir = test_save_dir + "svg_" + signature + "_ref/"
    svg_ref_list = os.listdir(svg_ref_dir)
    svg_ref_dict = {}

    svg_ref_img_dir = test_save_dir + "svg_" + signature + "_ref_img/"
    rm_mk_dir(svg_ref_img_dir)

    svg_ref_cairo_dir = test_save_dir + "svg_" + signature + "_ref_cairo/"
    rm_mk_dir(svg_ref_cairo_dir)

    for svg_ref_fn in svg_ref_list:
        svg_fp = svg_ref_dir + svg_ref_fn
        if (os.path.isdir(svg_fp)):
            continue

        shapes, shape_groups, point_var, color_var = load_init_svg(
            svg_fp,
            canvas_size=gt_rsz,
            trainable_stroke=False,
            requires_grad=False,
            experiment_dir=svg_ref_img_dir,
            svg_cario_dir=svg_ref_cairo_dir,
            add_circle=False,
            save_diffvg_img=True)

        svg_cairo_fp = svg_ref_cairo_dir + svg_ref_fn
        svg_ref_dict[svg_cairo_fp] = {
            "shapes": shapes,
            "shape_groups": shape_groups,
            "point_var": point_var,
            "color_var": color_var
        }

    svg_ref_img_seg_dir = test_save_dir + "svg_" + signature + "_ref_img_seg/"
    rm_mk_dir(svg_ref_img_seg_dir)

    svg_path_mask_dir = test_save_dir + "svg_" + signature + "_ref_path/"
    rm_mk_dir(svg_path_mask_dir)

    svg_sub_path_dir = test_save_dir + "svg_" + signature + "_sub_path/"
    rm_mk_dir(svg_sub_path_dir)

    svg_path_mask_zidx_dict = {}
    svg_img_path_match_dict = {}

    for svg_ref_key in svg_ref_dict.keys():
        print("svg_ref_key = ", svg_ref_key)

        shapes = svg_ref_dict[svg_ref_key]["shapes"]
        shape_groups = svg_ref_dict[svg_ref_key]["shape_groups"]
        point_var = svg_ref_dict[svg_ref_key]["point_var"]
        color_var = svg_ref_dict[svg_ref_key]["color_var"]

        svg_ref_key_fn_pre, _ = os.path.splitext(svg_ref_key.split("/")[-1])

        # svg path的mask image
        tmp_svg_path_mask_dir = svg_path_mask_dir + svg_ref_key_fn_pre + "/"
        rm_mk_dir(tmp_svg_path_mask_dir)

        tmp_svg_sub_path_dir = svg_sub_path_dir + svg_ref_key_fn_pre + "/"
        rm_mk_dir(tmp_svg_sub_path_dir)

        h = gt_rsz[0]
        w = gt_rsz[1]
        p_cnt = 0

        tmp_svg_path_mask_alpha_list = []
        ref_path_mask_fp_list = []
        for ini_path in shapes:
            # --------------------------------------
            tp_fill_color = color_var[p_cnt]
            tp_canvas_height = gt_rsz[0]
            tp_canvas_width = gt_rsz[1]

            tp_svg_path_fp = tmp_svg_sub_path_dir + str(p_cnt) + ".svg"
            save_path_svg(ini_path,
                          svg_path_fp=tp_svg_path_fp,
                          fill_color=tp_fill_color,
                          canvas_height=tp_canvas_height,
                          canvas_width=tp_canvas_width)

            # --------------------------------------

            img = path2img(ini_path, p_cnt, h, w, color_var)

            tmp_path_mask_alpha = (img.cpu()[:, :, 3] > 0).numpy()

            corrd_arr = np.where(tmp_path_mask_alpha)

            if (np.sum(corrd_arr[0]) == 0 or np.sum(corrd_arr[1]) == 0):
                tmp_svg_path_mask_alpha_list.append(np.array([]))
                p_cnt += 1
                os.remove(tp_svg_path_fp)
                continue

            # -----------------------------------------------
            t_mask_alpha = np.ascontiguousarray(tmp_path_mask_alpha * 255,
                                                dtype=np.uint8)

            mask_alpha_contour_max, mask_alpha_contour_max_bbox = find_max_contour_box(
                t_mask_alpha)

            if ((mask_alpha_contour_max is None)
                    or (mask_alpha_contour_max_bbox is None)):
                tmp_svg_path_mask_alpha_list.append(np.array([]))
                p_cnt += 1
                os.remove(tp_svg_path_fp)
                continue

            tmp_svg_path_mask_alpha_list.append(tmp_path_mask_alpha)

            # -----------------------------------------------

            img_rgba = PIL.Image.fromarray(
                (img.cpu().numpy() * 255).astype(np.uint8), "RGBA")
            tmp_svg_path_rgba_fp = tmp_svg_path_mask_dir + str(p_cnt) + ".png"
            img_rgba.save(tmp_svg_path_rgba_fp)
            ref_path_mask_fp_list.append(tmp_svg_path_rgba_fp)

            msk_x_mean = round(np.mean(corrd_arr[0]))
            msk_y_mean = round(np.mean(corrd_arr[1]))

            svg_path_mask_zidx_dict[tmp_svg_path_rgba_fp] = {
                "svg_path_mask_fp": tmp_svg_path_rgba_fp,
                "svg_cairo_fp": svg_ref_key,
                "z_idx": p_cnt,
                "msk_mean_xy": [msk_x_mean, msk_y_mean]
            }
            p_cnt += 1

        for spmzd_key in svg_path_mask_zidx_dict.keys():
            if (svg_path_mask_zidx_dict[spmzd_key]["svg_cairo_fp"] ==
                    svg_ref_key):
                svg_path_mask_zidx_dict[spmzd_key]["max_z_idx"] = p_cnt - 1

        svg_ref_path_match_dir = test_save_dir + "svg_" + \
            signature + "_ref_path_match/" + svg_ref_key_fn_pre + "/"
        rm_mk_dir(svg_ref_path_match_dir)

        ref_mask_dir = svg_ref_img_dir[:-1] + \
            "_mask/" + svg_ref_key_fn_pre + "/"
        rm_mk_dir(ref_mask_dir)

        for rfp_i in range(len(ref_path_mask_fp_list)):
            ref_m_fp_i = ref_path_mask_fp_list[rfp_i]
            ref_m_i_fn = ref_m_fp_i.split("/")[-1]
            ref_m_i_fn_pre = ref_m_i_fn.split(".")[0]

            ref_img_mask_rgba = PIL.Image.open(ref_m_fp_i)
            ref_img_mask_rgba = np.array(ref_img_mask_rgba)

            ref_img_mask = (ref_img_mask_rgba[:, :, 3] > 0)

            path_img_mask_fp = ref_m_fp_i

            for rfp_j in range(rfp_i+1, len(ref_path_mask_fp_list)):
                ref_m_fp_j = ref_path_mask_fp_list[rfp_j]
                mask_rgba_j = PIL.Image.open(ref_m_fp_j)
                mask_rgba_j = np.array(mask_rgba_j)  # (224, 224, 4)
                mask_j = (mask_rgba_j[:, :, 3] > 0)

                mask_j_sum = mask_j.sum()
                if (mask_j_sum == 0):
                    continue

                tmp_img_mask = ref_img_mask * mask_j  # [0, 1]
                sum_tmp_img_mask = tmp_img_mask.sum()
                if (sum_tmp_img_mask > 0):
                    ref_img_mask = ref_img_mask * (1-tmp_img_mask)

            ref_img_mask_rgba[:, :, 3] = ref_img_mask * 255

            ref_img_mask_rgba_sv_fp = ref_mask_dir + ref_m_i_fn_pre + "_ovlp.png"
            ref_img_mask_rgba_sv = PIL.Image.fromarray(
                ref_img_mask_rgba.astype(np.uint8), "RGBA")
            ref_img_mask_rgba_sv.save(ref_img_mask_rgba_sv_fp)

            plt.clf()
            fig, ax = plt.subplots(1,
                                   2,
                                   figsize=(5, 5),
                                   sharex=True,
                                   sharey=True)
            ax[0].imshow(ref_img_mask_rgba.astype(np.uint8))
            ax[0].set_title("cur_img_mask")

            path_img_mask_rgba = PIL.Image.open(path_img_mask_fp)
            ax[1].imshow(path_img_mask_rgba)
            ax[1].set_title('svg_path')
            svg_ref_path_match_fp = svg_ref_path_match_dir + ref_m_i_fn
            plt.savefig(svg_ref_path_match_fp)
            plt.close()

            svg_img_path_match_dict[ref_img_mask_rgba_sv_fp] = {
                "svg_path_mask_zidx_dict":
                svg_path_mask_zidx_dict[path_img_mask_fp],
            }

        svg_ref_dict[svg_ref_key]["shapes"] = []
        svg_ref_dict[svg_ref_key]["shape_groups"] = []
        svg_ref_dict[svg_ref_key]["point_var"] = []
        svg_ref_dict[svg_ref_key]["color_var"] = []

    sv_json(svg_path_mask_zidx_dict,
            test_save_dir + "svg_" + signature + "_path_mask_zidx_dict.json")
    sv_json(svg_img_path_match_dict,
            test_save_dir + "svg_" + signature + "_img_path_match_dict.json")


if __name__ == "__main__":
    # python svg_mask_path_match.py --signature=animal

    parser = argparse.ArgumentParser()
    parser.add_argument("--signature", type=str, default="animal")
    args = parser.parse_args()
    svg_mask_path_match(signature=args.signature)
