import os
import argparse
import random
import PIL
import numpy as np
import torch
import json
import shutil
import yaml

import json_help
from utils_match import rm_mk_dir, args_str2bool

from path_seg_corres import path_seg_match
from path_replace import tmask_img_replace

from dino_match.models.extractor import ViTExtractor
from dino_match.models.corrs_seg_match import Corrs_match

from img_pregroup import imgs_dir_group
from utils_pregroup import toposort_path

import pydiffvg

pydiffvg_render = pydiffvg.RenderFunction.apply


def img_path_match(signature, tar_img_fp="", ref_img_fp="", match_opt=None, gt_rsz=(224, 224)):
    test_save_dir = "./test_svg_custom/test_" + signature + "/"
    # ------------------------------------------

    tar_img_dir = test_save_dir + "tar_" + signature + "_img/"

    tar_img_mask_dir = test_save_dir + "tar_" + signature + "_img_mask/"

    mask_sim_dir = test_save_dir + "tar_" + signature + "_mask_sim/"
    # rm_mk_dir(mask_sim_dir)

    tar_img_dir_ini_svg_dir = test_save_dir + "tar_" + signature + "_img_ini_svg/"
    os.makedirs(tar_img_dir_ini_svg_dir, exist_ok=True)

    tar_img_dir_ini_svg_dir_nopotrace = test_save_dir + \
        "tar_" + signature + "_img_ini_svg_nopotrace/"
    os.makedirs(tar_img_dir_ini_svg_dir_nopotrace, exist_ok=True)

    tar_img_dir_ini_svg_dir_nopotrace_inicolor = test_save_dir + \
        "tar_" + signature + "_img_ini_svg_nopotrace_inicolor/"
    os.makedirs(tar_img_dir_ini_svg_dir_nopotrace_inicolor, exist_ok=True)

    tar_mask_potrace_path_dir = test_save_dir + \
        "tar_" + signature + "_mask_potrace_path/"

    ref_mask_list = []
    ref_ovlp_mask_list = []

    svg_img_path_match_dict_fp = test_save_dir + \
        "svg_" + signature + "_img_path_match_dict.json"
    with open(svg_img_path_match_dict_fp, encoding="utf-8") as f:
        svg_img_path_match_dict = json.load(f)
    # print("svg_img_path_match_dict = ", svg_img_path_match_dict)

    for spk in svg_img_path_match_dict.keys():
        # ref_mask_list.append(spk)
        ref_ovlp_mask_list.append(spk)

    svg_path_mask_zidx_dict_fp = test_save_dir + \
        "svg_" + signature + "_path_mask_zidx_dict.json"
    with open(svg_path_mask_zidx_dict_fp, encoding="utf-8") as f:
        svg_path_mask_zidx_dict = json.load(f)

    potrace_path_zidx_dict_fp = test_save_dir + "svg_" + \
        signature + "_potrace_path_zidx_dict.json"
    with open(potrace_path_zidx_dict_fp, encoding="utf-8") as f:
        potrace_path_zidx_dict = json.load(f)

    circle_mask_fp = ""
    for spk in svg_path_mask_zidx_dict.keys():
        ref_mask_list.append(spk)

        if (len(circle_mask_fp) == 0):
            tmp_max_z_idx = svg_path_mask_zidx_dict[spk]["max_z_idx"]
            circle_mask_dir = os.path.dirname(
                svg_path_mask_zidx_dict[spk]["svg_path_mask_fp"])
            circle_mask_fp = circle_mask_dir + "/" + str(
                tmp_max_z_idx) + ".png"

    im_fn = tar_img_fp.split("/")[-1]
    im_pre = im_fn.split(".")[0]

    tar_img_fp = tar_img_dir + im_fn

    tar_image = PIL.Image.open(tar_img_fp).convert(
        'RGB').resize(gt_rsz, PIL.Image.Resampling.BICUBIC)
    rgb_a_pil = tar_image
    rgb_a_np = np.array(rgb_a_pil)

    # ------------------------------------------
    img_mask = tar_img_mask_dir + im_pre + "/"
    img_mask_list = os.listdir(img_mask)
    new_img_mask_fp_list = []
    for im_msk_fp in img_mask_list:
        if (os.path.isdir(im_msk_fp)):
            continue
        if im_msk_fp.find("_mer") != -1:
            continue
        new_img_mask_fp_list.append(im_msk_fp)

    img_mask_list = new_img_mask_fp_list

    tar_img_mask_ovlp_dir = test_save_dir + "tar_" + \
        signature + "_img_mask_ovlp/" + im_pre + "/"
    tar_img_mask_ovlp_name_list = os.listdir(tar_img_mask_ovlp_dir)
    new_img_mask_ovlp_fp_list = []
    for im_msk_ovlp_fp in tar_img_mask_ovlp_name_list:
        if (os.path.isdir(im_msk_ovlp_fp)):
            continue

        if im_msk_ovlp_fp.find("_mer") != -1:
            continue

        new_img_mask_ovlp_fp_list.append(im_msk_ovlp_fp)

    tar_img_mask_ovlp_name_list = new_img_mask_ovlp_fp_list
    # ------------------------------------------

    cur_tar_img_svg_path_info = []
    mask_sim_sub_dir = mask_sim_dir + im_pre + "/"
    rm_mk_dir(mask_sim_sub_dir)

    tmp_tar_img_sim_info = {}

    TARGET_IMAGE_PATH = tar_img_fp
    tar_seg_dir = img_mask
    tar_seg_dir = tar_img_mask_ovlp_dir

    src_path_dir = ""

    cur_ref_img_pre = ref_img_fp.split("/")[-1].split(".")[0]

    for ovlp_fp in ref_ovlp_mask_list:
        dir_list = ovlp_fp.split('/')
        if dir_list[-2] == cur_ref_img_pre:
            src_path_dir = os.path.dirname(ovlp_fp) + '/'
            break

    aff_I2_weight = match_opt["aff_I2_weight"]
    aff_temp_ts = match_opt["aff_temp_ts"]
    aff_temp_qt = match_opt["aff_temp_qt"]
    judge_conn = match_opt["judge_conn"]

    tgt_img_segs_dict = path_seg_match(model_src=aff_model_src, model_tgt=aff_model_tgt, TARGET_IMAGE_PATH=TARGET_IMAGE_PATH, tar_seg_dir=tar_seg_dir, ref_img_fp=ref_img_fp,
                                       src_path_dir=src_path_dir, seg_files=tar_img_mask_ovlp_name_list, I2_weight=aff_I2_weight, temp_ts=aff_temp_ts, temp_qt=aff_temp_qt, judge_conn=judge_conn, signature=signature)

    # print("tgt_img_segs_dict:", tgt_img_segs_dict)
    for tgti in range(len(tgt_img_segs_dict)):
        tmp_tgt_seg_info = tgt_img_segs_dict[tgti]
        if (len(tmp_tgt_seg_info["path_info"]) == 0):
            continue

        tmp_seg_fp = tmp_tgt_seg_info["seg_fp"]
        tmp_match_path_fp = tmp_tgt_seg_info["match_path_fp"]
        # print("tmp_seg_fp = ", tmp_seg_fp)
        # print("tmp_match_path_fp = ", tmp_match_path_fp)

        tmp_tar_img_sim_info[tmp_seg_fp] = {
            "whole_path_img_mask_fp": tmp_match_path_fp,
            # "whole_max_I1_attn_sc_mean": whole_max_I1_attn_sc_mean.item(),
        }

    ovl_svg_path_mask_dict = {}
    for timk in tmp_tar_img_sim_info.keys():
        whole_path_img_mask_fp = tmp_tar_img_sim_info[timk]["whole_path_img_mask_fp"]
        if (whole_path_img_mask_fp in ovl_svg_path_mask_dict):
            ovl_svg_path_mask_dict[whole_path_img_mask_fp].append(timk)
        else:
            ovl_svg_path_mask_dict[whole_path_img_mask_fp] = [timk]

    mer_cnt = 0
    for ovlk in ovl_svg_path_mask_dict.keys():
        # print("ovlk = ", ovlk)
        if (len(ovl_svg_path_mask_dict[ovlk]) > 1):

            merge_mask = None
            for sfp in ovl_svg_path_mask_dict[ovlk]:
                s_img_mask_rgba = PIL.Image.open(sfp)
                s_img_mask_rgba = np.array(s_img_mask_rgba)
                s_img_mask = (s_img_mask_rgba[:, :, 3] > 0)

                if (merge_mask is None):
                    merge_mask = s_img_mask
                else:
                    merge_mask = merge_mask | s_img_mask

                tmp_tar_img_sim_info.pop(sfp)
                sfp_fn = sfp.split("/")[-1]
                img_mask_list.remove(sfp_fn)

            merge_mask = merge_mask.astype(np.uint8) * 255
            # add merge_mask to rgb_a_np, (224, 224, 3) to (224, 224, 4)
            ini_rgb_a = np.zeros(
                (rgb_a_np.shape[0], rgb_a_np.shape[1], 4), dtype=np.uint8)
            ini_rgb_a[:, :, 0:3] = rgb_a_np

            ini_rgb_a[:, :, 3] = merge_mask

            merge_mask = ini_rgb_a

            merge_mask_pil = PIL.Image.fromarray(merge_mask)
            mer_msk_fp = tar_img_mask_dir + im_pre + "/" + \
                im_pre + "_mer_" + str(mer_cnt) + ".png"
            merge_mask_pil.save(mer_msk_fp)

            # tmp_tar_img_sim_info.pop(sfp)
            tmp_tar_img_sim_info[mer_msk_fp] = {
                "whole_path_img_mask_fp": ovlk,
            }

            mer_msk_fn = mer_msk_fp.split("/")[-1]
            img_mask_list.append(mer_msk_fn)

            mer_cnt += 1

    # print("mer_cnt = ", mer_cnt)

    tmp_tar_img_path_info = tmask_img_replace(
        tmp_tar_img_sim_info, signature, gt_rsz)

    for sub_img_m in img_mask_list:
        sub_im_fp = img_mask + sub_img_m
        if (os.path.isdir(sub_im_fp)):
            continue

        sub_img_m_pre = os.path.splitext(sub_img_m)[0]

        if ((sub_im_fp in tmp_tar_img_path_info.keys()) and tmp_tar_img_path_info[sub_im_fp]["can_affine"]):
            cur_tar_img_svg_path_info.append({
                "tar_svg_path_mask_sub_fp": tmp_tar_img_path_info[sub_im_fp]["tar_svg_path_mask_sub_fp"],
                "tar_svg_path_sub_fp": tmp_tar_img_path_info[sub_im_fp]["tar_svg_path_sub_fp"],
                "fill_color_target": tmp_tar_img_path_info[sub_im_fp]["fill_color_target"],
                "cur_path_ini_color": tmp_tar_img_path_info[sub_im_fp]["cur_path_ini_color"],
            })

        else:
            potrace_path_dir = tar_mask_potrace_path_dir + im_pre + "/"
            tp_svg_path_fp = potrace_path_dir + sub_img_m_pre + ".svg"
            tp_svg_path_mask_fp = potrace_path_dir + sub_img_m_pre + ".png"

            if (tp_svg_path_fp in potrace_path_zidx_dict.keys()):

                tmp_svg_img_path_info = potrace_path_zidx_dict[tp_svg_path_fp]

                fill_color_target = torch.FloatTensor(
                    tmp_svg_img_path_info["fill_color_target"])
                cur_tar_img_svg_path_info.append({
                    "tar_svg_path_mask_sub_fp": tp_svg_path_mask_fp,
                    "tar_svg_path_sub_fp": tp_svg_path_fp,
                    "fill_color_target": fill_color_target
                })

    toposort_cur_tar_img_svg_path_info, cur_tar_shapes_ini, cur_tar_shapes_ini_groups, _, _, _, _ = toposort_path(
        cur_tar_img_svg_path_info, return_no_potrace=True)

    canvas_height = gt_rsz[0]
    canvas_width = gt_rsz[1]
    tar_img_dir_ini_svg_fp = tar_img_dir_ini_svg_dir + im_pre + '.svg'

    if (len(cur_tar_shapes_ini) > 0 and len(cur_tar_shapes_ini_groups) > 0):
        pydiffvg.save_svg(tar_img_dir_ini_svg_fp, canvas_width, canvas_height,
                          cur_tar_shapes_ini, cur_tar_shapes_ini_groups)


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python img_path_match.py --signature=animal --is_segment=1 --is_mask_match=1

    parser = argparse.ArgumentParser()
    parser.add_argument("--signature", type=str, default="animal")
    parser.add_argument("--is_segment", type=args_str2bool, default=True)
    parser.add_argument("--is_mask_match", type=args_str2bool, default=True)

    args = parser.parse_args()
    print("args.is_segment = ", args.is_segment)
    print("args.is_mask_match = ", args.is_mask_match)

    match_opt = json_help.parse("./img_path_match_param.yaml")
    match_opt = json_help.dict_to_nonedict(match_opt)

    if (args.is_segment):
        imgs_dir_group(signature=args.signature, pregroup_opt=match_opt)

    # Other constants
    PATH_TO_CONFIG = "./dino_match/config/default_config.yaml"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    COLORS = [[255, 0, 0], [255, 255, 0], [255, 0, 255],
              [0, 255, 0], [0, 0, 255], [0, 255, 255]]

    # Load arguments
    with open(PATH_TO_CONFIG) as f:
        aff_args = yaml.load(f, Loader=yaml.CLoader)

    aff_args['model_type'] = match_opt["aff_args_model_type"]
    aff_args['low_res_saliency_maps'] = match_opt["aff_args_low_res_saliency_maps"]
    aff_args['facet'] = match_opt["aff_args_facet"]
    aff_args['load_size'] = match_opt["aff_args_load_size"]
    aff_args['bin'] = match_opt["aff_args_bin"]
    aff_args['layer'] = match_opt["aff_args_layer"]
    aff_args['stride'] = match_opt["aff_args_stride"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model = ViTExtractor(aff_args['model_type'],
                              aff_args['stride'],
                              device=device)

    aff_model_src = Corrs_match(aff_args, dino_model)
    aff_model_tgt = Corrs_match(aff_args, dino_model)

    signature = args.signature
    test_save_dir = "./test_svg_custom/test_" + signature + "/"

    if (args.is_mask_match):

        ref_img_dir = test_save_dir + "svg_" + signature + "_ref_img/"
        ref_img_list = os.listdir(ref_img_dir)
        random.shuffle(ref_img_list)

        tar_img_dir = test_save_dir + "tar_" + signature + "_img/"
        tar_img_list = os.listdir(tar_img_dir)
        random.shuffle(tar_img_list)

        for tar_img_fn in tar_img_list:

            tar_img_fp = tar_img_dir + tar_img_fn
            if (os.path.isdir(tar_img_fp)):
                continue

            tar_img_fn_pre = tar_img_fn.split(".")[0]

            ref_img_fp = ""
            for ref_img_fn in ref_img_list:
                tmp_ref_img_fp = ref_img_dir + ref_img_fn
                if (os.path.isdir(tmp_ref_img_fp)):
                    continue

                ref_img_fn_pre = ref_img_fn.split(".")[0]
                if (len(ref_img_fn_pre) > 0 and (ref_img_fn_pre in tar_img_fn_pre)):
                    ref_img_fp = ref_img_dir + ref_img_fn
                    break

            if (ref_img_fp == ""):
                continue

            print("tar_img_fp = ", tar_img_fp)
            print("ref_img_fp = ", ref_img_fp)

            img_path_match(signature=args.signature,
                           tar_img_fp=tar_img_fp, ref_img_fp=ref_img_fp, match_opt=match_opt)

    # -----------------------------------------------
    tar_img_dir_ini_svg_dir = test_save_dir + "tar_" + signature + "_img_ini_svg/"

    aft_process_dir = test_save_dir + "tar_" + signature + "_img_ini_svg_aftp/"
    rm_mk_dir(aft_process_dir)

    svg_pa_dir = tar_img_dir_ini_svg_dir
    svg_pa_list = os.listdir(tar_img_dir_ini_svg_dir)

    for idr in svg_pa_list:
        svg_fp = os.path.join(svg_pa_dir, idr)
        if (os.path.isdir(svg_fp)):
            continue

        idr_pre = idr.split(".")[0]

        shutil.copy(svg_fp, aft_process_dir)

    del_dir = test_save_dir + "tar_" + signature + "_mask_replace/"
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = test_save_dir + "tar_" + signature + "_out/"
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = test_save_dir + "tar_" + signature + "_mask_sim/"
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = test_save_dir + "tar_" + signature + "_mask_potrace_path/"
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = test_save_dir + "tar_" + signature + "_img_mask_ovlp/"
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
    del_dir = test_save_dir + "tar_" + signature + "_img_seg/"
    if (os.path.exists(del_dir)):
        shutil.rmtree(del_dir)
