import networkx as nx
import os
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Vision imports
import torch
import cv2

import copy

from dino_match.models.correspondence_functions import (
    overlay_segment, resize)


from dino_match.models.corrs_seg_match import find_part_correspondences_seg


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLORS = [[255, 0, 0], [255, 255, 0], [255, 0, 255],
          [0, 255, 0], [0, 0, 255], [0, 255, 255]]


def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def show_mask(mask, ax, color=None):
    if color is None:
        # color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        color = np.array([204/255, 203/255, 203/255, 1.0])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def hide_axes(axs):
    for ax in axs:
        ax.axis('off')


def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def find_dominant_color(image_array, color_space="lab", k=3):

    reshaped_image = image_array.reshape(-1, 3)

    if (color_space == "lab"):
        lab_pixels = cv2.cvtColor(np.float32(reshaped_image.reshape(
            1, -1, 3)) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)

    else:
        lab_pixels = reshaped_image

    k = min(k, lab_pixels.shape[0])
    kmeans_img = KMeans(n_clusters=k, max_iter=500, n_init=5).fit(lab_pixels)
    kmeans_labels = kmeans_img.labels_

    counts = np.bincount(kmeans_labels)
    dominant_color = kmeans_img.cluster_centers_[np.argmax(counts)]

    return dominant_color


def rgb_to_lab(color):
    color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    lab_color = cv2.cvtColor(color / 255.0, cv2.COLOR_RGB2LAB)
    return lab_color.reshape(3)


def color_difference(color1, color2):
    return np.sqrt(np.sum((color1 - color2) ** 2))


def is_similar(color1, color2, threshold, color_space='lab'):
    difference = color_difference(color1, color2)
    return difference < threshold


def find_connected_components(mask):

    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))

    connected_components = np.zeros((num_labels-1, 224, 224), dtype=np.uint8)

    for k in range(1, num_labels):
        connected_components[k-1] = (labels == k).astype(np.uint8) * 255

    return connected_components


def load_rgb(path, gt_rsz=(224, 224)):
    """ Loading RGB image with OpenCV
    : param path: string, image path name. Must point to a file.
    """
    assert os.path.isfile(path), f"Path {path} doesn't exist"

    # return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    target = PIL.Image.open(path)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB").resize(gt_rsz, PIL.Image.Resampling.BICUBIC)
    # img = np.array(target).astype("float")
    return target


def path_seg_match(model_src, model_tgt, TARGET_IMAGE_PATH, tar_seg_dir, ref_img_fp, src_path_dir, seg_files=None,
                   I2_weight=1.0, temp_ts=0.02, temp_qt=0.2, load_size=(224, 224), judge_conn=True, signature="many"):

    if (seg_files is None):
        seg_files = os.listdir(tar_seg_dir)

    # Load the target image
    rgb_b = load_rgb(TARGET_IMAGE_PATH, load_size)
    # rgb_b = Image.fromarray(rgb_b)
    # rgb_b = PIL.Image.open(TARGET_IMAGE_PATH).convert('RGB')
    # rgb_b = rgb_b.resize(load_size, PIL.Image.Resampling.BICUBIC)
    tar_img_pre = TARGET_IMAGE_PATH.split("/")[-1].split(".")[0]

    img_pil = PIL.Image.open(TARGET_IMAGE_PATH).resize(
        load_size, PIL.Image.Resampling.BICUBIC)

    img_pil_np = np.array(img_pil)
    if (img_pil_np.shape[2] == 4):
        rgb_b_mask = img_pil_np[:, :, 3]
    else:
        rgb_b_mask = np.ones_like(img_pil_np[:, :, 0])

    rgb_b_mask = (np.array(rgb_b_mask, dtype=np.int32) > 0)

    img_segs_dict = []
    for seg_fn in seg_files:
        seg_fp = os.path.join(tar_seg_dir, seg_fn)
        if (not os.path.isfile(seg_fp)):
            continue

        seg_mask_rgba = np.array(Image.open(seg_fp))
        seg_mask = (seg_mask_rgba[:, :, 3] > 0)

        fg_overlap = rgb_b_mask * seg_mask
        overlap_thresh = 0.8
        if (np.sum(fg_overlap) / np.sum(seg_mask) > overlap_thresh):
            seg_mask_sum = np.sum(seg_mask)
            img_segs_dict.append({
                "seg_fp": seg_fp,
                "seg_mask": seg_mask,
                "seg_mask_sum": seg_mask_sum
            })

    img_segs_dict = sorted(
        img_segs_dict, key=lambda x: x["seg_mask_sum"], reverse=True)

    model_tgt.set_image(
        rgb_b, img_segs_dict=img_segs_dict, fg_mask=rgb_b_mask)
    model_tgt.generate_clusters_seg()

    rgb_a = load_rgb(ref_img_fp)

    path_files = os.listdir(src_path_dir)

    # ----------------------------------------------------------------
    new_path_files = []
    for path_fn in path_files:
        path_fp = os.path.join(src_path_dir, path_fn)
        if (not os.path.isfile(path_fp)):
            continue
        new_path_files.append(path_fn)

    new_path_files = sorted(new_path_files, key=lambda x: int(
        x.split(".")[0].split("_")[0]))
    path_files = new_path_files
    # ----------------------------------------------------------------

    src_parts = []
    src_fp_list = []
    for path_fn in path_files:
        path_fp = os.path.join(src_path_dir, path_fn)
        if (not os.path.isfile(path_fp)):
            continue
        path_mask_rgba = np.array(Image.open(path_fp))
        path_mask = (path_mask_rgba[:, :, 3] > 0)
        src_parts.append(path_mask)
        src_fp_list.append(path_fp)
        # print("path_fp = ", path_fp)

    model_src.set_image(rgb_a, img_segs_dict=None, fg_mask=None)
    model_src.generate_clusters_query(query_parts=src_parts)

    src_query_parts = src_parts
    src_descs = model_src.get_clu_info("descs")
    K_queries_desc = model_src.get_clu_info("K_queries_desc")
    K_query_parts = model_src.get_clu_info("K_query_parts")
    query_parts_idx = model_src.get_clu_info("query_parts_idx")
    src_nps = model_src.get_clu_info("num_patches")
    # N_Kq = model_src.get_clu_info("N_Kq")
    N_Kq_list = model_src.get_clu_info("N_Kq_list")

    K_target_desc = model_tgt.get_clu_info("K_target_desc")
    K_target_mapped = model_tgt.get_clu_info("K_target_mapped")
    tgt_img_resized = model_tgt.get_clu_info("img_resized")
    tgt_load_size = model_tgt.get_clu_info("load_size")
    tgt_nps = model_tgt.get_clu_info("num_patches")
    N_Kt = model_tgt.get_clu_info("N_Kt")
    tgt_load_shape = model_tgt.get_clu_info("load_shape")

    parts_out, I1_maps = find_part_correspondences_seg(src_descs=src_descs, src_query_parts=K_query_parts, K_queries_desc=K_queries_desc,
                                                       K_target_desc=K_target_desc, K_target_mapped=K_target_mapped, tgt_img_resized=tgt_img_resized, tgt_load_size=tgt_load_size,
                                                       src_nps=src_nps, tgt_nps=tgt_nps, N_Kq_list=N_Kq_list, N_Kt=N_Kt,
                                                       tgt_load_shape=tgt_load_shape, I2_weight=I2_weight, temp_ts=temp_ts, temp_qt=temp_qt)

    # print("parts_out.shape = ", parts_out.shape)
    # parts_out.shape =  (1, 256, 256)
    tgt_img_segs_dict = model_tgt.get_clu_info("img_segs_dict")

    rgb_a = np.array(rgb_a)
    rgb_b = np.array(rgb_b)
    quer_img = rgb_a.copy().astype(np.uint8)
    corr_img = rgb_b.copy().astype(np.uint8)
    exist_cnt = 0

    test_save_dir = "./test_svg_custom/test_" + signature + "/"

    out_dir = test_save_dir + "dino_match/" + tar_img_pre + "/"
    os.makedirs(out_dir, exist_ok=True)

    lab_threshold = 10.0
    rgb_threshold = 50

    path_conn_match_list = []

    for idx, part_i in enumerate(src_query_parts):
        if idx not in query_parts_idx:
            continue

        i = exist_cnt
        exist_cnt += 1
        # print("i = ", i)

        sum_parts_out_ini = parts_out[i].sum()
        if (sum_parts_out_ini == 0):
            continue

        part_out_i_ini = resize(parts_out[i], corr_img.shape[:2]) > 0

        parts_out_ini = np.uint8(parts_out[i] > 0)
        parts_out_conn = find_connected_components(parts_out_ini)
        # print("parts_out_conn.shape = ", parts_out_conn.shape)

        if (judge_conn == False):
            parts_out_conn = np.expand_dims(parts_out_ini, axis=0)
        # ----------------------------------------------------------------

        real_query_parts_idx_cnt = 0
        for pck, parts_conn in enumerate(parts_out_conn):
            sum_parts_conn = parts_conn.sum() / 255.0
            if (sum_parts_conn < 1):
                continue

            real_query_parts_idx_cnt += 1

        for pck, parts_conn in enumerate(parts_out_conn):

            sum_parts_conn = parts_conn.sum() / 255

            if (sum_parts_conn < 1):
                continue

            # ----------------------------------------------------------------
            part_out_i = resize(parts_conn, corr_img.shape[:2]) > 0

            I1_attnmap = I1_maps[i]

            mer_mask = np.zeros(
                (part_out_i.shape[0], part_out_i.shape[1]), dtype=bool)
            mer_seg_cnt = 0
            overlap_thresh = 0.5

            candi_selected_seg_indices = []

            for pi in range(len(tgt_img_segs_dict)):
                seg_mask = tgt_img_segs_dict[pi]["seg_mask"]

                sum_seg_mask = np.sum(seg_mask)
                if (sum_seg_mask == 0):
                    continue

                parts_overlap = part_out_i * seg_mask
                if (np.sum(parts_overlap) / np.sum(seg_mask) > overlap_thresh):
                    mer_seg_cnt += 1
                    candi_selected_seg_indices.append(pi)

                    mer_mask = mer_mask | seg_mask

            I1_attn_sc_mean = 0

            if (np.sum(mer_mask) > 0):
                tgt_img = rgb_b.copy().astype(np.uint8)

                mer_mask_pixels = tgt_img[mer_mask]

                dominant_mer_mask_lab = find_dominant_color(
                    mer_mask_pixels, k=10)

                selected_seg_indices = []
                new_mer_mask = np.zeros(
                    (part_out_i.shape[0], part_out_i.shape[1]), dtype=bool)
                for pi in range(len(tgt_img_segs_dict)):

                    if (pi not in candi_selected_seg_indices):
                        continue

                    if ("path_info" not in tgt_img_segs_dict[pi]):
                        tgt_img_segs_dict[pi]["path_info"] = []

                    seg_mask = tgt_img_segs_dict[pi]["seg_mask"]

                    tgt_img = rgb_b.copy().astype(np.uint8)
                    seg_mask_pixels = tgt_img[seg_mask]
                    dominant_seg_mask_lab = find_dominant_color(
                        seg_mask_pixels, k=3)

                    color_sim_flg = is_similar(
                        dominant_seg_mask_lab, dominant_mer_mask_lab, threshold=lab_threshold)

                    if (color_sim_flg == False):
                        continue

                    selected_seg_indices.append(pi)
                    new_mer_mask = new_mer_mask | seg_mask

                # mer_mask = mer_mask & color_mask
                mer_mask = new_mer_mask

                mer_mask_rsz = resize(mer_mask, I1_attnmap.shape)
                sum_mer_mask_rsz = mer_mask_rsz.sum()
                if (sum_mer_mask_rsz > 0):
                    I1_attn_sc_mean = (
                        I1_attnmap * mer_mask_rsz).sum() / sum_mer_mask_rsz

                    path_conn_match_list.append(
                        {"query_parts_idx": idx, "query_parts_idx_cnt": real_query_parts_idx_cnt, "mer_mask": mer_mask.copy(), "I1_attn_sc_mean": I1_attn_sc_mean, "mer_mask_sum": mer_mask.sum(), "dominant_mer_mask_lab": copy.deepcopy(dominant_mer_mask_lab), "selected_seg_indices": copy.deepcopy(selected_seg_indices)})

            # ------------------------------------

    path_conn_match_list = sorted(
        path_conn_match_list,
        key=lambda x: (x["query_parts_idx_cnt"], -x["I1_attn_sc_mean"], -x["mer_mask_sum"]), reverse=False
    )

    path_idx_used_list = []
    mer_mask_used = np.zeros(
        (corr_img.shape[0], corr_img.shape[1]), dtype=bool)

    for path_conn_match_itm in path_conn_match_list:
        query_parts_idx = path_conn_match_itm["query_parts_idx"]

        if (query_parts_idx not in path_idx_used_list):
            # print("query_parts_idx = ", query_parts_idx)

            mer_mask = path_conn_match_itm["mer_mask"].copy()
            dominant_mer_mask_lab = copy.deepcopy(
                path_conn_match_itm["dominant_mer_mask_lab"])

            I1_attn_sc_mean = path_conn_match_itm["I1_attn_sc_mean"]
            mer_mask_sum = path_conn_match_itm["mer_mask_sum"]
            selected_seg_indices = path_conn_match_itm["selected_seg_indices"]

            # ------------------------------------

            mer_mask_used_ovlp = mer_mask_used & mer_mask
            mer_mask_used_ovlp_sum = np.sum(mer_mask_used_ovlp)
            mer_mask_used_ovlp_thresh = 0.8

            if (mer_mask_sum == 0 or (mer_mask_used_ovlp_sum / mer_mask_sum > mer_mask_used_ovlp_thresh)):
                continue

            mer_mask_used = mer_mask_used | mer_mask
            # ------------------------------------

            path_idx_used_list.append(query_parts_idx)
            # ------------------------------------

            for pi in range(len(tgt_img_segs_dict)):
                if ("path_info" not in tgt_img_segs_dict[pi]):
                    tgt_img_segs_dict[pi]["path_info"] = []

                if (pi not in selected_seg_indices):
                    continue

                tgt_img_segs_dict[pi]["path_info"].append(
                    {"query_parts_idx": query_parts_idx, "I1_attn_sc_mean": I1_attn_sc_mean})

            print("Selected seg_mask indices:", selected_seg_indices)

    path_seg_mask_dict = {}

    for pi in range(len(tgt_img_segs_dict)):
        tmp_path_info = tgt_img_segs_dict[pi]["path_info"]

        if (len(tmp_path_info) > 0):
            tmp_path_info = sorted(
                tmp_path_info, key=lambda x: x["I1_attn_sc_mean"], reverse=True)
            tgt_img_segs_dict[pi]["path_info"] = tmp_path_info

            cur_query_parts_idx = tmp_path_info[0]["query_parts_idx"]
            cur_query_path_fp = src_fp_list[cur_query_parts_idx]
            tgt_img_segs_dict[pi]["match_path_fp"] = cur_query_path_fp

            # ------------------------------------
            if (cur_query_parts_idx not in path_seg_mask_dict):
                path_seg_mask_dict[cur_query_parts_idx] = []

            path_seg_mask_dict[cur_query_parts_idx].append(
                tgt_img_segs_dict[pi]["seg_mask"].copy())
            # ------------------------------------

        tgt_img_segs_dict[pi]["seg_mask"] = []

        ini_seg_fp = tgt_img_segs_dict[pi]["seg_fp"]

        new_seg_fp = ini_seg_fp.replace('_ovlp', '')
        tgt_img_segs_dict[pi]["seg_fp"] = new_seg_fp

    # ------------------------------------

    new_path_cnt = len(src_parts) + 1

    cluster_label_img = np.ones((corr_img.shape[0], corr_img.shape[1])) * (-1)
    cluster_label_img_list = []

    for query_parts_idx in path_seg_mask_dict:
        seg_mask_list = path_seg_mask_dict[query_parts_idx]
        cur_merge_mask = np.zeros(seg_mask_list[0].shape, dtype=bool)
        for seg_mask_i in seg_mask_list:
            cur_merge_mask = cur_merge_mask | seg_mask_i

        cluster_label_img[cur_merge_mask] = query_parts_idx
        cluster_label_img_list.append(
            {"query_parts_idx": query_parts_idx, "cur_merge_mask": cur_merge_mask})

        # ------------------------------------

    # --------------------------------------------
    plt.clf()
    plt.figure()
    plt.imshow(rgb_a)

    cmap = 'nipy_spectral'
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, new_path_cnt + 1))

    sorted_cluster_label_img_list = sorted(
        cluster_label_img_list, key=lambda x: x["query_parts_idx"])

    tar_bg_mask = np.ones((corr_img.shape[0], corr_img.shape[1]), dtype=bool)

    for item in sorted_cluster_label_img_list:
        mask = item["cur_merge_mask"]
        tar_bg_mask[mask] = False
        show_mask(mask, plt.gca(), color=colors[item["query_parts_idx"]])

    show_mask(tar_bg_mask, plt.gca())

    plt.close()
    # --------------------------------------------

    # --------------------------------------------
    plt.clf()
    plt.figure()
    plt.imshow(rgb_a)

    cmap = 'nipy_spectral'
    cmap = plt.get_cmap(cmap)

    ref_bg_mask = np.ones((quer_img.shape[0], quer_img.shape[1]), dtype=bool)
    for si in range(len(src_parts)):
        mask = src_parts[si]
        ref_bg_mask[mask] = False
        show_mask(mask, plt.gca(), color=colors[si])

    show_mask(ref_bg_mask, plt.gca())

    plt.close()
    # --------------------------------------------

    return tgt_img_segs_dict
