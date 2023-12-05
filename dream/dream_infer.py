import os
import random
import argparse
from skimage.transform import resize
import numpy as np
import shutil
import PIL
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
import diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.optimization import get_scheduler

from rembg import remove, new_session
import clip
from clip_loss import m_clip_text_loss
from prompt_utils import get_negtive_prompt_text, get_prompt_text


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def remove_background(img_pil, out_fp):

    # u2net, isnet-general-use, isnet-anime, sam
    model_name = "sam"
    session = new_session(model_name)

    img_fg_mask = remove(img_pil, only_mask=True)

    # 0-255
    img_fg_mask_norm = np.array(img_fg_mask, dtype=np.float32) / 255.0

    img_fg_mask_norm[img_fg_mask_norm < 0.5] = 0
    img_fg_mask_norm[img_fg_mask_norm >= 0.5] = 1

    img_fg_mask_ini = img_fg_mask_norm.astype(np.uint8)
    # img_fg_mask_ini.shape: (512, 512)

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
    img_with_alpha.save(out_fp)

    return img_with_alpha


def dream_infer(prompt, base_negprompt, model_id, num_samples):
    # 50
    BASE_STEPS = 70
    # 7.5
    BASE_SCALE = 8

    pipe_test = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")

    # pipe_test.safety_checker = lambda images, clip_input: (images, False)
    pipe_test.safety_checker = lambda images, clip_input: (images, None)

    # 'ViT-B/32'
    clip_model, preprocess = clip.load('ViT-B/16', "cuda", jit=False)

    all_images = []
    # 用于保存图像和相应的tmp_clip_loss值
    img_loss_pairs = []
    images = pipe_test(prompt,
                       negative_prompt=base_negprompt,
                       num_images_per_prompt=num_samples,
                       num_inference_steps=BASE_STEPS,
                       guidance_scale=BASE_SCALE).images

    # save images in pred_dir
    for i, ini_img in enumerate(images):
        # 将PIL图像转换为PyTorch张量
        img = ini_img.resize((224, 224))
        transform = ToTensor()
        img_tensor = transform(img)

        # 添加批量维度
        img_tensor = img_tensor.unsqueeze(0)

        tmp_prompt = prompt.split("(")[0]

        tmp_clip_loss = m_clip_text_loss(
            img_tensor, tmp_prompt, 4, clip_model)

        # 将图像和tmp_clip_loss添加到列表中
        img_loss_pairs.append((ini_img, tmp_clip_loss.item()))

    # 根据tmp_clip_loss对图像进行排序，最小值在前
    img_loss_pairs.sort(key=lambda x: x[1])

    # 从排序后的元组列表中提取排序后的图像
    sorted_images = [pair[0] for pair in img_loss_pairs]

    # 将排序后的图像扩展到all_images列表中
    all_images.extend(sorted_images)
    print("len_all_images = ", len(all_images))

    return all_images


def dream_infer_multi(concept_n, prompt_description):

    num_samples = 3
    num_rows = 5

    concept_dir = "./dream_concept/"

    dream_pred_dir = "./dream_pred/"
    # if os.path.exists(dream_pred_dir):
    #     shutil.rmtree(dream_pred_dir)

    base_negprompt = get_negtive_prompt_text()

    signature = concept_n

    model_id = os.path.join(concept_dir, concept_n + "_dream_concept")

    concept_pred_dir = os.path.join(dream_pred_dir, signature)

    concept_pred_prompt_dir = os.path.join(
        concept_pred_dir, prompt_description)
    os.makedirs(concept_pred_prompt_dir, exist_ok=True)

    prompt = get_prompt_text(
        signature, description_start="a clipart of ", description_end=" " + prompt_description + ", ")

    print("prompt = ", prompt)

    all_images = []
    # num_remain = int(num_samples/2)
    num_remain = int(num_samples)
    for sd_i in range(num_rows):
        tmp_images = dream_infer(prompt=prompt,
                                 base_negprompt=base_negprompt,
                                 model_id=model_id,
                                 num_samples=num_samples)

        all_images.extend(tmp_images[:num_remain])

    for i, img in enumerate(all_images):
        tmp_fp = os.path.join(concept_pred_prompt_dir,
                              f"{signature}_{prompt_description}_dream{i}.png")
        img.save(tmp_fp)

    grid = image_grid(all_images, num_rows, num_remain)
    tmp_fp = os.path.join(concept_pred_prompt_dir, signature + ".png")
    grid.save(tmp_fp)

    # if the target image contains background, it's better to mask it out
    pred_img_list = os.listdir(concept_pred_prompt_dir)
    for i_fn in pred_img_list:
        i_fp = os.path.join(concept_pred_prompt_dir, i_fn)

        if os.path.isdir(i_fp):
            shutil.rmtree(i_fp)
            continue
        else:
            if (not i_fn.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm',
                    '.tif', '.tiff'))):
                os.remove(i_fp)
                continue

        # 已经去除过背景的图像不再mask
        i_fn_pre = i_fn.split(".")[0]
        if (i_fn_pre.startswith("masked_") or (not "dream" in i_fn_pre)):
            continue

        target = Image.open(i_fp)
        out_fp = os.path.join(
            concept_pred_prompt_dir, "masked_" + i_fn)

        remove_background(img_pil=target, out_fp=out_fp)


if __name__ == "__main__":
    # python dream_infer.py --concept_n 03670_animal_8 --prompt_description "wearing a top hat"

    parser = argparse.ArgumentParser(description="Process concept and prompt.")
    parser.add_argument("--concept_n", type=str,
                        help="The name of the concept directory")
    parser.add_argument("--prompt_description", type=str,
                        help="Description for the prompt")
    args = parser.parse_args()

    concept_dir = "./dream_concept/"

    # Check if the specified concept directory exists
    concept_path = os.path.join(
        concept_dir, args.concept_n + "_dream_concept")
    if not os.path.exists(concept_path):
        print("Concept directory does not exist.")
    else:
        dream_infer_multi(args.concept_n, args.prompt_description)
