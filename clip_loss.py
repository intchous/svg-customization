import collections
import clip
import torch
import torch.nn as nn
from torchvision import models, transforms
from munch import DefaultMunch
from ssim_loss import SSIM
from iou_loss import IOU
from multiscale_loss import gaussian_pyramid_loss


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        args = {
            "attention_init": 1,
            "aug_scale_min": 0.7,
            "augemntations": 'affine',
            "augment_both": 1,
            "batch_size": 1,
            "clip_conv_layer_weights": [0.0, 0.0, 1.0, 1.0, 0.0],

            "clip_conv_loss": 1,
            # "clip_conv_loss_type": 'L2',
            "clip_conv_loss_type": 'Cos',

            "clip_fc_loss_weight": 0.1,
            "clip_model_name": 'RN101',
            "clip_text_guide": 0,
            "clip_weight": 0,
            "color_lr": 0.01,
            "color_vars_threshold": 0.0,
            "control_points_per_seg": 4,
            "device": 'cuda',
            "display": 0,
            "display_logs": 0,
            "eval_interval": 10,
            "fix_scale": 0,
            "force_sparse": 0,
            "image_scale": 224,
            "include_target_in_aug": 0,
            "lr": 1.0,
            "lr_scheduler": 0,
            "mask_object": 0,
            "mask_object_attention": 0,
            "noise_thresh": 0.5,

            # "num_aug_clip": 4,
            "num_aug_clip": 10,

            "num_iter": 501,
            "num_paths": 32,
            "num_segments": 1,
            "num_stages": 1,
            "output_dir":
            './CLIPasso/output_sketches/blue/blue_32strokes_seed2000',
            "path_svg": 'none',
            "percep_loss": 'none',
            "perceptual_weight": 0,
            "saliency_clip_model": 'ViT-B/32',
            "saliency_model": 'clip',
            "save_interval": 10,
            "seed": 2000,
            "softmax_temp": 0.3,
            "start_clip": 0,
            "target":
            './CLIPasso/target_images/blue.png',
            "text_target": 'none',
            "train_with_clip": 0,
            "use_gpu": 1,
            "use_wandb": 0,
            "wandb_name": 'blue_32strokes_seed2000',
            "wandb_project_name": 'none',
            "wandb_user": 'yael-vinker',
            "width": 1.5,
            "xdog_intersec": 1
        }
        args = DefaultMunch.fromDict(args)

        self.args = args
        self.percep_loss = args.percep_loss

        self.train_with_clip = args.train_with_clip
        self.clip_weight = args.clip_weight
        self.start_clip = args.start_clip

        self.clip_conv_loss = args.clip_conv_loss
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.clip_text_guide = args.clip_text_guide

        self.losses_to_apply = self.get_losses_to_apply()

        self.loss_mapper = \
            {
                "clip": CLIPLoss(args),
                "clip_conv_loss": CLIPConvLoss(args),
                "l2": L2_(args),
                "lpips": LPIPS(args),
                "ssim": SSIM_(args),
                "iou": IOU_(args)
            }

    def get_losses_to_apply(self):
        losses_to_apply = []
        # losses_to_apply.append("l2")
        # losses_to_apply.append("lpips")
        # losses_to_apply.append("ssim")
        # losses_to_apply.append("iou")
        losses_to_apply.append("clip_conv_loss")

        # if self.percep_loss != "none":
        #     losses_to_apply.append(self.percep_loss)
        # if self.train_with_clip and self.start_clip == 0:
        #     losses_to_apply.append("clip")
        # if self.clip_conv_loss:
        #     losses_to_apply.append("clip_conv_loss")
        # if self.clip_text_guide:
        #     losses_to_apply.append("clip_text")

        return losses_to_apply

    def update_losses_to_apply(self, epoch):
        if "clip" not in self.losses_to_apply:
            if self.train_with_clip:
                if epoch > self.start_clip:
                    self.losses_to_apply.append("clip")

    # (sketches, inputs.detach(), renderer.get_color_parameters(), renderer, counter, optimizer)
    def forward(
            self,
            sketches,
            targets,
            # color_parameters,
            # renderer,
            epoch,
            # points_optim=None,
            mode="train"):
        loss = 0
        self.update_losses_to_apply(epoch)

        losses_dict = dict.fromkeys(self.losses_to_apply,
                                    torch.tensor([0.0]).to(self.args.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)

        # loss_coeffs["clip"] = self.clip_weight
        # loss_coeffs["clip_text"] = self.clip_text_guide
        # loss_coeffs["l2"] = 0.1
        # loss_coeffs["ssim"] = 0.05
        # loss_coeffs["lpips"] = 0.05

        for loss_name in self.losses_to_apply:
            if loss_name in ["clip_conv_loss"]:
                conv_loss = self.loss_mapper[loss_name](sketches, targets,
                                                        mode)
                for layer in conv_loss.keys():
                    losses_dict[layer] = conv_loss[layer]
            elif loss_name == "l2":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets).mean()
            # elif loss_name == "ssim":
            #     losses_dict[loss_name] = 1.0 - self.loss_mapper[loss_name](
            #         sketches, targets)
            else:
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
            # loss = loss + self.loss_mapper[loss_name](sketches, targets).mean() * loss_coeffs[loss_name]

        for key in self.losses_to_apply:
            # loss = loss + losses_dict[key] * loss_coeffs[key]
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]

        return losses_dict


class CLIPLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPLoss, self).__init__()

        self.args = args
        self.model, clip_preprocess = clip.load('ViT-B/32',
                                                args.device,
                                                jit=False)
        self.model.eval()
        self.preprocess = transforms.Compose([clip_preprocess.transforms[-1]
                                              ])  # clip normalisation
        self.device = args.device
        self.NUM_AUGS = args.num_aug_clip
        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(
                transforms.RandomPerspective(fill=0,
                                             p=1.0,
                                             distortion_scale=0.5))
            augemntations.append(
                transforms.RandomResizedCrop(224,
                                             scale=(0.8, 0.8),
                                             ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.calc_target = True
        self.include_target_in_aug = args.include_target_in_aug
        self.counter = 0
        self.augment_both = args.augment_both

    def forward(self, sketches, targets, mode="train"):
        if self.calc_target:
            targets_ = self.preprocess(targets).to(self.device)
            self.targets_features = self.model.encode_image(targets_).detach()
            self.calc_target = False

        if mode == "eval":
            # for regular clip distance, no augmentations
            with torch.no_grad():
                sketches = self.preprocess(sketches).to(self.device)
                sketches_features = self.model.encode_image(sketches)
                return 1. - torch.cosine_similarity(sketches_features,
                                                    self.targets_features)

        loss_clip = 0
        sketch_augs = []
        img_augs = []
        for n in range(self.NUM_AUGS):
            augmented_pair = self.augment_trans(torch.cat([sketches, targets]))
            sketch_augs.append(augmented_pair[0].unsqueeze(0))

        sketch_batch = torch.cat(sketch_augs)
        # sketch_utils.plot_batch(img_batch, sketch_batch, self.args, self.counter, use_wandb=False, title="fc_aug{}_iter{}_{}.jpg".format(1, self.counter, mode))
        # if self.counter % 100 == 0:
        # sketch_utils.plot_batch(img_batch, sketch_batch, self.args, self.counter, use_wandb=False, title="aug{}_iter{}_{}.jpg".format(1, self.counter, mode))

        sketch_features = self.model.encode_image(sketch_batch)

        for n in range(self.NUM_AUGS):
            loss_clip += (1. - torch.cosine_similarity(
                sketch_features[n:n + 1], self.targets_features, dim=1))
        self.counter += 1
        return loss_clip
        # return 1. - torch.cosine_similarity(sketches_features, self.targets_features)


class LPIPS(torch.nn.Module):
    def __init__(self,
                 args,
                 pretrained=True,
                 normalize=True,
                 pre_relu=True,
                 device=None):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        if (device == None):
            device = args.device
        self.normalize = normalize
        self.pretrained = pretrained
        augemntations = []
        augemntations.append(
            transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(
            transforms.RandomResizedCrop(224,
                                         scale=(0.8, 0.8),
                                         ratio=(1.0, 1.0)))
        self.augment_trans = transforms.Compose(augemntations)
        self.feature_extractor = LPIPS._FeatureExtractor(pretrained,
                                                         pre_relu).to(device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]

        # TODO(mgharbi) Apply Richard's linear weights?

        if self.normalize:
            diffs = [torch.sum((p - t)**2, 1) for (p, t) in zip(pred, target)]
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t)**2, 1) for (p, t) in zip(pred, target)]

        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        return sum(diffs)

    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            vgg_pretrained = models.vgg16(pretrained=pretrained).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer(
                "shift",
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                "scale",
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift) / self.scale
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats


class L2_(torch.nn.Module):
    def __init__(self, args):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(L2_, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        augemntations = []
        augemntations.append(
            transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(
            transforms.RandomResizedCrop(224,
                                         scale=(0.8, 0.8),
                                         ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        # LOG.warning("LPIPS is untested")

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        # Get VGG features

        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)
        diffs = [torch.square(p - t).mean() for (p, t) in zip(pred, target)]
        return sum(diffs)


class SSIM_(torch.nn.Module):
    def __init__(self, args):
        super(SSIM_, self).__init__()
        augemntations = []
        augemntations.append(
            transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(
            transforms.RandomResizedCrop(224,
                                         scale=(0.8, 0.8),
                                         ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.ssim_lo = SSIM()

    def forward(self, pred, target, mode="train"):
        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)

        diffs = [(1.0 - self.ssim_lo(pred, target))
                 for (p, t) in zip(pred, target)]
        # return sum(diffs)
        return sum(diffs) * 1.0 / len(diffs)


class IOU_(torch.nn.Module):
    def __init__(self, args):
        super(IOU_, self).__init__()
        augemntations = []
        augemntations.append(
            transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5))
        augemntations.append(
            transforms.RandomResizedCrop(224,
                                         scale=(0.8, 0.8),
                                         ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.iou_lo = IOU()

    def forward(self, pred, target, mode="train"):
        sketch_augs, img_augs = [pred], [target]
        if mode == "train":
            for n in range(4):
                augmented_pair = self.augment_trans(torch.cat([pred, target]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        pred = torch.cat(sketch_augs, dim=0)
        target = torch.cat(img_augs, dim=0)

        diffs = [self.iou_lo(pred, target) for (p, t) in zip(pred, target)]
        return sum(diffs) * 1.0 / len(diffs)


class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        # print("clip_model = ", clip_model)
        self.featuremaps = None

        # 12 resblocks in VIT visual transformer
        for i in range(12):
            self.clip_model.visual.transformer.resblocks[
                i].register_forward_hook(self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [
        torch.square(x_conv - y_conv).mean()
        for x_conv, y_conv in zip(xs_conv_features, ys_conv_features)
    ]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [
        torch.abs(x_conv - y_conv).mean()
        for x_conv, y_conv in zip(xs_conv_features, ys_conv_features)
    ]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    # if "RN" in clip_model_name:
    #     return [
    #         torch.square(x_conv, y_conv, dim=1).mean()
    #         for x_conv, y_conv in zip(xs_conv_features, ys_conv_features)
    #     ]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean()
            for x_conv, y_conv in zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = args.clip_model_name
        # self.clip_model_name = "ViT-B/32"
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2",
            "Cos",
            "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2",
            "Cos",
            "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(self.clip_model_name,
                                                args.device,
                                                jit=False)

        self.model = torch.load('./models/RN101.pt')

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            # print("layers = ", layers)
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(
                transforms.RandomPerspective(fill=0,
                                             p=1.0,
                                             distortion_scale=0.5))
            augemntations.append(
                transforms.RandomResizedCrop(224,
                                             scale=(0.8, 0.8),
                                             ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)
                                 ], [self.normalize_transform(y)]
        x_augs = [x]
        gt_augs = [y]

        if mode == "train":
            for n in range(self.num_augs):
                augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))
                x_augs.append(augmented_pair[0].unsqueeze(0))
                gt_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        pym_loss = 0
        x_pred = torch.cat(x_augs, dim=0)
        gt_target = torch.cat(gt_augs, dim=0)

        for n in range(self.num_augs):
            pym_loss += gaussian_pyramid_loss(x_pred[n:n + 1],
                                              gt_target[n:n + 1])

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        # self.args.clip_conv_layer_weights = [8.0, 6.0, 4.0, 4.0, 0.0]
        self.args.clip_conv_layer_weights = [8.0, 6.0, 4.0, 2.0, 0.0]

        # "clip_conv_layer_weights": [0.0, 0.0, 1.0, 1.0, 0.0],
        # clip_conv_layer_weights = [1.0, 1.0, 1.0, 1.0, 0],
        # clip_conv_layer_weights = [0, 0, 0, 0, 1.0],

        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[
                    f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(
                xs_fc_features, ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        pym_loss_weight = 10.0
        conv_loss_dict["pym_loss"] = pym_loss * pym_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2),
                             (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x

        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]
