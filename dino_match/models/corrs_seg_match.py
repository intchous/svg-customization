import gc
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Skimage
from skimage.transform import resize
from skimage import img_as_bool

# Pytorch
import torch
from pytorch_metric_learning import distances
import torchvision.transforms as tf

# Custom imports
from .extractor import ViTExtractor
from .crf import CRF
from .clustering import get_K_means_v2
from .correspondence_functions import (get_load_shape,
                                       image_resize,
                                       get_saliency_maps_V2,
                                       get_saliency_maps_V3,
                                       map_descriptors_to_clusters,
                                       rescale_pts,
                                       uv_im_to_desc,
                                       uv_desc_to_im,
                                       map_values_to_segments,
                                       )


def to_int(x):
    """ Returns an integer type object
    : param x: torch.Tensor, np.ndarray or scalar.
    """
    if torch.is_tensor(x):
        return x.long()
    if isinstance(x, np.ndarray):
        return x.astype(np.int32)
    return int(x)


class Corrs_match(object):
    def __init__(self, args, model=None):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = ViTExtractor(self.args['model_type'],
        #                           self.args['stride'],
        #                           device=self.device)
        self.model = model
        self.d_fn = distances.CosineSimilarity()
        return

    def _cleanup(self):
        """ Private cleanup action """
        torch.cuda.empty_cache()
        gc.collect()

    def set_image(self, im, img_segs_dict=None, fg_mask=None):
        """ Set the image and parts.
        :param im: path to, or PIL.Image
        img_segs_dict: [{
                "seg_fp": seg_fp,
                "seg_mask": seg_mask,
                "seg_mask_sum": seg_mask_sum
            }]
        :param fg_mask: [Optional] np.ndarray, Foreground mask. If None
                          is passed, the mask is based on saliency map.
        """
        self._cleanup()
        if isinstance(im, str):
            im = Image.open(im)

        if not isinstance(im, Image.Image):
            raise TypeError("Wrong type of input im - must be PIL.Image")

        self.img = im
        self.fg_mask = fg_mask
        self.img_segs_dict = img_segs_dict
        self.preprocess()
        return

    def preprocess(self):
        """ Prepares descriptors, clusters, etc.
        """
        with torch.no_grad():
            load_size = self.args["load_size"]
            # self.load_size = load_size

            self.load_shape = get_load_shape(self.img, load_size)

            self.img_resized = image_resize(
                self.img, [load_size, None])

            if (self.img_segs_dict is not None):
                # resize 到load_size (eg. 256*256)
                for pi in range(len(self.img_segs_dict)):
                    tmp_seg = self.img_segs_dict[pi]["seg_mask"]
                    rsz_load = img_as_bool(resize(tmp_seg, self.load_shape))
                    self.img_segs_dict[pi]["seg_mask"] = rsz_load

            self.img_batch, _ = self.model.preprocess_image(
                self.img_resized, self.load_shape)

            self.descs = self.model.extract_descriptors(
                self.img_batch.to(self.device),
                self.args["layer"],
                self.args["facet"],
                self.args["bin"]).cpu()

            self.num_patches = self.model.num_patches
            self.load_size = self.model.load_size

            if (self.fg_mask is not None):
                # 比较resize和interpolate的效果
                nps = self.num_patches
                # self.fg_mask_rsz = resize(self.fg_mask, nps).astype(np.bool)
                self.fg_mask_rsz = resize(self.fg_mask, nps).astype(bool)

            """ Get saliency mask """
            # saliency_map.shape =  (256, 256)
            # 0: background, 1: foreground
            # self.saliency_mask = self.fg_mask

        return

    def generate_clusters_query(self, query_parts=[], N_Kq_ratio=0.2, N_Kq_bar=10):
        """ Generate query clusters
        :param N_Kq: parameter for number of query clusters
        """
        # self.N_Kq = N_Kq

        nps = self.num_patches
        # nps =  (63, 63)
        # self.descs.shape =  torch.Size([1, 1, 3969, 6528])

        self.desc_image_tensor = self.descs.reshape(
            nps[0], nps[1], -1)

        self.desc_image = self.desc_image_tensor.numpy()

        self.K_queries_desc = []
        self.K_query_parts = []
        self.query_parts_idx = []
        self.N_Kq_list = []
        for i, part in enumerate(query_parts):
            part = img_as_bool(resize(part, self.load_shape))
            q_a = resize(part, nps).astype(bool)
            # part.shape =  (256, 256)
            # q_a.shape =  (63, 63)

            # extract query K-means descriptors
            query_desc = self.desc_image[q_a]

            if (query_desc.shape[0] == 0):
                continue

            # Attempting K=10
            # N_Kq = min(N_Kq, query_desc.shape[0])
            # self.N_Kq = N_Kq
            # tmp_N_Kq = min(N_Kq_bar, query_desc.shape[0])

            if (query_desc.shape[0] <= N_Kq_bar):
                tmp_N_Kq = query_desc.shape[0]
                K_query = query_desc
            else:
                tmp_N_Kq = max(N_Kq_bar, int(query_desc.shape[0] * N_Kq_ratio))

                K_query, labels = get_K_means_v2([query_desc[None, None, ...]], 1,
                                                 self.args["elbow"],
                                                 list(range(tmp_N_Kq, tmp_N_Kq+1)))

                print("K_query.shape = ", K_query.shape)
                # K_query.shape =  (10, 6528)

            self.N_Kq_list.append(tmp_N_Kq)
            self.K_queries_desc.append(K_query)
            self.K_query_parts.append(part)
            self.query_parts_idx.append(i)

        return

    def generate_clusters_seg(self):
        nps = self.num_patches
        self.desc_image_tensor = self.descs.reshape(
            nps[0], nps[1], -1)

        self.desc_image = self.desc_image_tensor.numpy()

        whole_mask = np.zeros(
            (self.load_shape[0], self.load_shape[1]), dtype=bool)

        self.segs_desc_single = None

        cluster_label_img = np.ones((nps[0], nps[1])) * (-1)
        real_cluster_cnt = 0

        assert self.img_segs_dict is not None
        for pi in range(len(self.img_segs_dict)):
            seg = self.img_segs_dict[pi]["seg_mask"]

            seg_rsz = resize(seg, nps).astype(bool)
            seg_rsz_sum = np.sum(seg_rsz)
            if (seg_rsz_sum < 1):
                continue

            whole_mask = whole_mask | seg

            cluster_label_img[seg_rsz] = real_cluster_cnt
            real_cluster_cnt += 1

            tmp_seg_desc = self.desc_image[seg_rsz]

            K_seg = tmp_seg_desc.mean(axis=0, keepdims=True)

            # ------------------------------
            if (self.segs_desc_single is None):
                self.segs_desc_single = K_seg
            else:
                self.segs_desc_single = np.concatenate(
                    (self.segs_desc_single, K_seg), axis=0)
            # ------------------------------

        whole_mask_pil = Image.fromarray(whole_mask.astype(np.uint8)*255)
        # whole_mask_pil.save("whole_mask.png")
        # --------------------------------------------

        # --------------------------------------------
        # cmap = 'nipy_spectral'
        cmap = 'jet'
        cmap = "Blues_r"
        cmap = "Greens_r"
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(cluster_label_img.astype(np.uint8),
                  vmin=0, vmax=real_cluster_cnt, cmap=cmap)
        # fig.savefig('./clustering_before.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # --------------------------------------------

        # --------------------------------------------
        assert self.fg_mask_rsz is not None
        cluster_label_img[(1 - self.fg_mask_rsz).astype(bool)
                          ] = real_cluster_cnt

        foreground_area = cluster_label_img == -1

        foreground_desc = self.desc_image[foreground_area]
        similarities = np.dot(foreground_desc, self.segs_desc_single.T)
        max_similarities = np.argmax(similarities, axis=1)

        for idx, foreground_coord in enumerate(zip(*np.nonzero(foreground_area))):
            cluster_label_img[foreground_coord] = max_similarities[idx]

        cluster_label_img = cluster_label_img.astype(int)
        # --------------------------------------------

        # --------------------------------------------
        # cluster_label_img = C.cpu().numpy()
        # cmap = 'jet' if (real_cluster_cnt > 10) else 'tab10'
        # cmap = 'nipy_spectral'
        cmap = 'jet'
        cmap = "Blues_r"
        cmap = "Greens_r"
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(cluster_label_img.astype(np.uint8),
                  vmin=0, vmax=real_cluster_cnt, cmap=cmap)
        # fig.savefig('./clustering_aft.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # --------------------------------------------

        self.N_Kt = real_cluster_cnt + 1
        self.K_target_desc = self.segs_desc_single
        self.K_target_mapped = cluster_label_img
        return

    def generate_clusters_pixel(self, N_Kt=200):
        nps = self.num_patches
        self.desc_image_tensor = self.descs.reshape(
            nps[0], nps[1], -1)

        self.desc_image = self.desc_image_tensor.numpy()
        # print("self.desc_image.shape = ", self.desc_image.shape)
        # self.desc_image.shape =  (63, 63, 6528)

        assert self.fg_mask_rsz is not None
        fg_desc = self.desc_image[self.fg_mask_rsz == True]
        # print("fg_desc.shape = ", fg_desc.shape)
        # fg_desc.shape =  (556, 6528)

        # Attempting N_Kt
        N_Kt = min(N_Kt, fg_desc.shape[0])
        # N_Kt = fg_desc.shape[0]

        K_target, K_labels = get_K_means_v2([fg_desc[None, None, ...]], 1,
                                            self.args["elbow"],
                                            list(range(N_Kt, N_Kt+1)))

        cluster_label_img = np.ones((nps[0], nps[1])) * (-1)
        cluster_label_img[self.fg_mask_rsz == True] = K_labels.cpu().numpy()

        cluster_label_img[(1 - self.fg_mask_rsz).astype(bool)] = N_Kt
        # print("cluster_label_img = ", cluster_label_img)

        # --------------------------------------------
        # cmap = 'nipy_spectral'
        cmap = 'jet'
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(cluster_label_img.astype(np.uint8),
                  vmin=0, vmax=N_Kt, cmap=cmap)
        # fig.savefig('./clustering_before.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # --------------------------------------------

        self.N_Kt = N_Kt + 1
        self.K_target_desc = K_target
        self.K_target_mapped = cluster_label_img
        return

    def get_clu_info(self, attrib_name=""):
        if (attrib_name == "descs"):
            return self.descs

        elif (attrib_name == "num_patches"):
            return self.num_patches
        elif (attrib_name == "load_shape"):
            load_shape = self.img_batch[0, 0].shape
            return load_shape
        elif (attrib_name == "load_size"):
            return self.load_size
        elif (attrib_name == "img_resized"):
            return self.img_resized

        elif (attrib_name == "K_queries_desc"):
            return self.K_queries_desc
        elif (attrib_name == "K_query_parts"):
            return self.K_query_parts
        elif (attrib_name == "query_parts_idx"):
            return self.query_parts_idx
        # elif(attrib_name == "N_Kq"):
        #     return self.N_Kq

        elif (attrib_name == "N_Kq_list"):
            return self.N_Kq_list

        elif (attrib_name == "N_Kt"):
            return self.N_Kt
        elif (attrib_name == "K_target_desc"):
            return self.K_target_desc
        elif (attrib_name == "K_target_mapped"):
            return self.K_target_mapped

        elif (attrib_name == "img_segs_dict"):
            return self.img_segs_dict

        else:
            assert False, "There is no attrib_name = {}".format(attrib_name)


def build_affcorrs(version=1, **kwargs):
    """ Get AffCorrs model based on version """
    if version == 1:
        return Corrs_match(kwargs)
    else:
        raise NotImplementedError("Requested version is not implemented yet")


def find_part_correspondences_seg(src_descs=None, src_query_parts=None, K_queries_desc=None,
                                  K_target_desc=None, K_target_mapped=None, tgt_img_resized=None, tgt_load_size=None,
                                  src_nps=None, tgt_nps=None, N_Kq_list=[], N_Kt=200,
                                  tgt_load_shape=None, I2_weight=1.0,
                                  temp_ts=0.02, temp_qt=0.2, d_fn=distances.CosineSimilarity()):
    """ Find part correspondences
    : param temp_ts: float, Target to support temperature
    : param temp_qt: float, Query to target temperature
    """

    # Mapping from Target centroids to Source descriptors
    A_ts = d_fn(src_descs[0, 0], torch.Tensor(K_target_desc)).T  # K2, H1.W1

    # Transform to probability: Softmax along HW axis (HW,K)
    # Probability of each cluster belonging a particular area of the source image.
    P_ts = torch.nn.Softmax(dim=1)(A_ts/temp_ts)  # [K2, H1.W2]
    # print("P_ts.shape = ", P_ts.shape)
    # P_ts.shape =  torch.Size([80, 3969])
    P_ts_img = P_ts.reshape(-1, src_nps[0], src_nps[1])  # [K2, H1, W2]

    # Output is [N parts, Target Size]
    segm_out = np.zeros((len(src_query_parts), *tgt_load_shape))

    I1_maps = []

    # print("len_src_query_parts = ", len(src_query_parts))
    for i, part in enumerate(src_query_parts):
        # Get similarity from K1 to K2
        # self.K_queries_desc[i].shape =  (10, 6528)
        # [K1, K2]
        A_qt = d_fn(torch.Tensor(
            K_queries_desc[i]), torch.Tensor(K_target_desc))
        # print("A_qt.shape = ", A_qt.shape)
        # A_qt.shape =  torch.Size([10, 80])

        # [K1, K2]
        P_qt = torch.nn.Softmax(dim=1)(A_qt/temp_qt)
        # P_qt.shape =  torch.Size([10, 80])

        # Calculate probability that each cluster in target image is matching
        # to the part mask in the source image.
        part_mask = resize(img_as_bool(part), src_nps).astype(bool)
        # part_mask.shape =  (63, 63)

        P_tq = P_ts_img[:, part_mask]
        # print("P_tq.shape = ", P_tq.shape)
        # P_tq.shape =  torch.Size([80, 336])
        P_tq = P_tq.sum(-1)  # [K2,]
        # P_tq.shape =  torch.Size([80])

        # Get scores for each cluster in Target
        # how likely is it a match to the query
        # And how likely does it match to the query
        S_fg = P_qt.sum(0) * P_tq
        # S_fg = P_qt.sum(0)

        I1 = map_values_to_segments(K_target_mapped, S_fg)

        I1_maps.append(I1)

        # ---------------------------------------------
        I1_mask = (I1[:, :] > 0.01)

        I1_show = I1
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow((I1_show.cpu().numpy() * 255.0).astype(np.uint8),
                  vmin=0, vmax=255, cmap='jet')
        # fig.savefig('./I1_map.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        # ---------------------------------------------

        tmp_N_Kq = N_Kq_list[i]
        I2 = torch.ones(tgt_nps)*(0.5 * tmp_N_Kq / N_Kt)  # 0.0625
        I2 = I2 * I2_weight

        # Set as Unary to CRF
        # Lower energy = bigger distance
        # Higher probability = lower distance.
        P = torch.stack([I1, I2], dim=-1).numpy()
        # P = torch.stack([I1, 1-I1], dim=-1).numpy()

        P = P.reshape(*tgt_nps, -1)

        final = CRF(tgt_img_resized, P,
                    tgt_nps, None, tgt_load_size)

        final = final.reshape(*tgt_load_shape)

        segm_out[i] = final.reshape(*tgt_load_shape) > 0

    return segm_out, I1_maps
