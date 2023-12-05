import os
import numpy as np
from pixel_to_svg import _merge_mean_color, _weight_mean_color

from skimage import segmentation
from skimage.future import graph
from skimage.segmentation import mark_boundaries
from skimage import segmentation

import PIL
import cv2
import graphlib
import pydiffvg
import torch

from rembg import remove

from utils_match import rm_mk_dir, flood_fill, find_max_contour_box, judge_mask


def get_segmentation(img_dir,
                     img_fn,
                     img_seg_dir,
                     gt_rsz=(224, 224),
                     seg_max_dist=20,
                     seg_ratio=0.5,
                     seg_kernel_size=3,
                     seg_sigma=0,
                     rag_sigma=255.0,
                     rag_connectivity=2,
                     rag_mode="distance",
                     mer_thresh=30,
                     is_padding=False,
                     is_save_seg_mask=True,
                     is_judge_mask=True,
                     is_flood_fill=False,
                     is_merge_seg=True,
                     img_fg_mask=None
                     ):
    infile = os.path.join(img_dir, img_fn)
    fn = img_fn
    im_pre, im_ext = os.path.splitext(fn)

    target = PIL.Image.open(infile)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB").resize(gt_rsz, PIL.Image.Resampling.BICUBIC)
    img = np.array(target).astype("float")

    if (is_padding):

        pd_row_l = 1
        pd_row_r = 1
        pd_col_l = 1
        pd_col_r = 1

        img = np.pad(img,
                     pad_width=[(pd_row_l, pd_row_r), (pd_col_l, pd_col_r),
                                (0, 0)],
                     mode='constant')

    seg_quickshift = segmentation.quickshift(img,
                                             ratio=seg_ratio,
                                             kernel_size=seg_kernel_size,
                                             max_dist=seg_max_dist,
                                             sigma=seg_sigma)

    if (img_fg_mask is not None):
        seg_quickshift[~img_fg_mask] = -1

    mb = mark_boundaries(img, seg_quickshift, color=(255, 255, 0))
    mb = PIL.Image.fromarray((mb).astype(np.uint8))
    mb.save(img_seg_dir + im_pre + "_quickshift_seg_ini.png")

    seg = seg_quickshift

    if (is_merge_seg):
        g = graph.rag_mean_color(
            img,
            seg,
            connectivity=rag_connectivity,
            mode=rag_mode,
            sigma=rag_sigma,
        )

        seg = graph.merge_hierarchical(seg,
                                       g,
                                       thresh=mer_thresh,
                                       rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=_merge_mean_color,
                                       weight_func=_weight_mean_color)

        mb = mark_boundaries(img, seg, color=(255, 255, 0))
        mb = PIL.Image.fromarray((mb).astype(np.uint8))
        mb.save(img_seg_dir + im_pre + "_seg_aft.png")

    nb_layers = None
    if len(seg.shape) == 2:
        if nb_layers is None:
            nb_layers = seg.max() + 1
        masks = np.zeros((seg.shape[0], seg.shape[1], nb_layers)).astype(bool)
        m = masks.reshape((-1, nb_layers))
        s = seg.reshape((-1, ))
        m[np.arange(len(m)), s] = 1
        assert np.all(masks.argmax(axis=2) == seg)
    else:
        masks = seg

    if (is_save_seg_mask):
        mask_dir = img_dir[:-1] + "_mask/" + im_pre + "/"
        rm_mk_dir(mask_dir)

    cond_masks = []
    m_cnt = 0
    # add an alpha channel to img (224, 224, 4)
    img = np.dstack((img, np.ones((img.shape[0], img.shape[1])) * 255))
    for layer in range(masks.shape[2]):
        mask = masks[:, :, layer]

        if (is_judge_mask and (judge_mask(mask=mask, gt_rsz=gt_rsz) == False)):
            # mask = np.zeros_like(mask)
            # cond_masks.append(mask)
            continue

        mask_repeat = np.expand_dims(mask, 2).repeat(4, axis=2)
        mask_img_rgba = img * mask_repeat

        mask_alpha = mask_img_rgba[:, :, 3]  # 255
        mask_alpha = np.ascontiguousarray(mask_alpha, dtype=np.uint8)

        if (is_flood_fill):
            mask_alpha_filld = flood_fill(mask_alpha,
                                          canvas_size=(mask_alpha.shape[0],
                                                       mask_alpha.shape[1]))
            mask_alpha = mask_alpha_filld
            # mask = (mask_alpha == 255)
            mask = (mask_alpha > 0)

        if (is_judge_mask):
            mask_alpha_contour_max, mask_alpha_contour_max_bbox = find_max_contour_box(
                mask_alpha)
            if ((mask_alpha_contour_max is None)
                    or (mask_alpha_contour_max_bbox is None)):
                continue

        if (is_save_seg_mask):
            mb = PIL.Image.fromarray((mask_img_rgba).astype(np.uint8), "RGBA")
            mb.save(mask_dir + im_pre + "_seg_" + str(m_cnt) + ".png")

        cond_masks.append(mask)
        m_cnt += 1

    # stack all masks
    cond_masks = np.stack(cond_masks, axis=2)
    # assert np.all(cond_masks == masks)
    if (is_judge_mask == False):
        assert (cond_masks.shape[2] == masks.shape[2])

    return img, cond_masks


# -------------------------------------------------


def toposort_path(cur_tar_img_svg_path_info, return_no_potrace=False):
    tpsort = graphlib.TopologicalSorter()

    for pzi in range(len(cur_tar_img_svg_path_info) - 1):
        p_info_pzi = cur_tar_img_svg_path_info[pzi]
        pzi_img_mask_rgba = PIL.Image.open(
            p_info_pzi["tar_svg_path_mask_sub_fp"])
        pzi_img_mask_rgba = np.array(pzi_img_mask_rgba)
        pzi_img_mask = (pzi_img_mask_rgba[:, :, 3] > 0)

        for pzj in range(pzi + 1, len(cur_tar_img_svg_path_info)):
            p_info_pzj = cur_tar_img_svg_path_info[pzj]
            pzj_img_mask_rgba = PIL.Image.open(
                p_info_pzj["tar_svg_path_mask_sub_fp"])
            pzj_img_mask_rgba = np.array(pzj_img_mask_rgba)
            # pzj_img_mask = (pzj_img_mask_rgba[:, :, 3] == 255)
            pzj_img_mask = (pzj_img_mask_rgba[:, :, 3] > 0)

            overlap_mask = pzi_img_mask * pzj_img_mask
            sum_overlap_mask = overlap_mask.sum()

            if (sum_overlap_mask == 0):
                continue

            area_prop_pzi = sum_overlap_mask * 1.0 / pzi_img_mask.sum()
            area_prop_pzj = sum_overlap_mask * 1.0 / pzj_img_mask.sum()

            if (area_prop_pzi > area_prop_pzj):
                # pzj 在 pzi 前面
                tpsort.add(pzi, pzj)
            else:
                # pzi 在 pzj 前面
                tpsort.add(pzj, pzi)

    tpsort_stk = [*tpsort.static_order()]
    # print("tpsort_stk = ", tpsort_stk)

    new_rank = []
    toposort_cur_tar_img_svg_path_info = []

    tp_cnt = 0
    for i in range(len(cur_tar_img_svg_path_info)):
        if (i in tpsort_stk):
            toposort_cur_tar_img_svg_path_info.append(
                cur_tar_img_svg_path_info[tpsort_stk[tp_cnt]])
            new_rank.append(tpsort_stk[tp_cnt])
            tp_cnt += 1
        else:
            toposort_cur_tar_img_svg_path_info.append(
                cur_tar_img_svg_path_info[i])
            new_rank.append(i)

    assert (tp_cnt == len(tpsort_stk))

    cur_tar_img_svg_path_info = toposort_cur_tar_img_svg_path_info

    # ------------------------------

    cur_tar_shapes_ini = []
    cur_tar_shapes_ini_groups = []

    cur_tar_shapes_ini_no_potrace = []
    cur_tar_shapes_ini_groups_no_potrace = []
    no_potrace_pz = 0

    cur_tar_shapes_ini_no_potrace_inicolor = []
    cur_tar_shapes_ini_groups_no_potrace_inicolor = []

    for pz in range(len(toposort_cur_tar_img_svg_path_info)):
        p_info = toposort_cur_tar_img_svg_path_info[pz]
        svg_aff_path_fp = p_info["tar_svg_path_sub_fp"]

        cur_w, cur_h, cur_shapes, cur_shape_groups = pydiffvg.svg_to_scene(
            svg_aff_path_fp)

        p_cnt = 0
        cur_path = None
        for path in cur_shapes:
            cur_path = path
            p_cnt += 1

        assert (p_cnt == 1)

        cur_tar_shapes_ini.append(cur_path)
        fill_color_init = p_info["fill_color_target"]

        # ---------------------------------------------
        if "cur_path_ini_color" in p_info:
            fill_color_init = p_info["cur_path_ini_color"]
        # ---------------------------------------------

        cur_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
            [pz]),
            fill_color=fill_color_init,
            use_even_odd_rule=False)
        cur_tar_shapes_ini_groups.append(cur_path_group)

        # ------------------------------
        if (not ("potrace" in svg_aff_path_fp)):
            cur_tar_shapes_ini_no_potrace.append(cur_path)
            cur_tar_shapes_ini_no_potrace_inicolor.append(cur_path)

            cur_path_group_no_potrace = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
                [no_potrace_pz]),
                fill_color=fill_color_init,
                use_even_odd_rule=False)
            cur_tar_shapes_ini_groups_no_potrace.append(
                cur_path_group_no_potrace)

            cur_path_ini_color = p_info["cur_path_ini_color"]
            cur_path_group_no_potrace_inicolor = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
                [no_potrace_pz]),
                fill_color=cur_path_ini_color,
                use_even_odd_rule=False)

            cur_tar_shapes_ini_groups_no_potrace_inicolor.append(
                cur_path_group_no_potrace_inicolor)

            no_potrace_pz += 1

        # ------------------------------

    if (return_no_potrace):
        return toposort_cur_tar_img_svg_path_info, cur_tar_shapes_ini, cur_tar_shapes_ini_groups, cur_tar_shapes_ini_no_potrace, cur_tar_shapes_ini_groups_no_potrace, cur_tar_shapes_ini_no_potrace_inicolor, cur_tar_shapes_ini_groups_no_potrace_inicolor
    else:
        return toposort_cur_tar_img_svg_path_info, cur_tar_shapes_ini, cur_tar_shapes_ini_groups

# -------------------------------------------------


# -------------------------------------------------
def canny_edge(image, low_threshold, high_threshold):
    # Normalize to 0-255 and convert to uint8
    norm_img = (image / np.max(image) * 255).astype(np.uint8)
    gray = cv2.cvtColor(norm_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def dilate_edges(edges, kernel_size):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return dilated_edges

# -------------------------------------------------


def get_fg_mask(img_pil):

    img_fg_mask = remove(img_pil, only_mask=True, post_process_mask=True)

    img_fg_mask_norm = np.array(img_fg_mask, dtype=np.float32) / 255.0

    img_fg_mask_norm[img_fg_mask_norm < 0.5] = 0
    img_fg_mask_norm[img_fg_mask_norm >= 0.5] = 1

    img_fg_mask_ini = img_fg_mask_norm.astype(np.uint8)

    return img_fg_mask_ini


def add_mask_to_image(img_pil, img_fg_mask_ini, out_fp=None):

    # Expand the mask and copy it to all 3 channels
    img_fg_mask = img_fg_mask_ini[:, :, np.newaxis]
    img_fg_mask = np.repeat(img_fg_mask, 3, axis=2)

    img = np.array(img_pil)
    img_fg = img * img_fg_mask

    # add alpha channel to img_fg
    img_alpha = np.concatenate(
        [img_fg, img_fg_mask_ini[:, :, np.newaxis] * 255], axis=2)
    img_with_alpha = PIL.Image.fromarray(
        img_alpha.astype(np.uint8), mode='RGBA')

    if (out_fp is not None):
        img_with_alpha.save(out_fp)

    return img_with_alpha

# -------------------------------------------------
