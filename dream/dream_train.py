import gc
import diffusers
from PIL import Image
import argparse
import itertools
import math
import os
import random
import requests
from io import BytesIO
from pathlib import Path
from accelerate.utils import set_seed
import shutil
from argparse import Namespace

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import option_utils


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# Setup and check the images you have just added
def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


# Setup the Classes
class DreamBoothDataset(Dataset):
    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            class_data_root=None,
            class_prompt=None,
            size=512,
            center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose([
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size)
            if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


# -----------------------------------------------------


def dream_train(signature, instance_img_dir, output_dir, dream_opt):

    # Settings for your newly created concept
    # `instance_prompt` is a prompt that should contain a good description of what your object or style is, together with the initializer word `cat_toy`
    instance_prompt = signature + " clipart"

    # ----------------------------------------------------
    pretrained_model_name_or_path = dream_opt["pretrained_model_name_or_path"]
    sample_batch_size = dream_opt["sample_batch_size"]
    learning_rate = dream_opt["learning_rate"]
    max_train_steps = dream_opt["max_train_steps"]
    train_batch_size = dream_opt["train_batch_size"]
    save_steps = dream_opt["save_steps"]
    Scheduler_type = dream_opt["Scheduler_type"]
    gradient_accumulation_steps = dream_opt["gradient_accumulation_steps"]
    lr_warmup_steps = dream_opt["lr_warmup_steps"]
    seed = dream_opt["seed"]
    train_text_encoder = dream_opt["train_text_encoder"]
    lr_scheduler = dream_opt["lr_scheduler"]

    # ----------------------------------------------------
    # delete files and directory in instance_img_dir if they are not images
    for file in os.listdir(instance_img_dir):
        tmp_fp = os.path.join(instance_img_dir, file)
        if os.path.isdir(tmp_fp):
            shutil.rmtree(tmp_fp)
        else:
            if (not file.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm',
                 '.tif', '.tiff'))):
                os.remove(tmp_fp)
    # ----------------------------------------------------

    # Check the `prior_preservation` option if you would like class of the concept (e.g.: toy, dog, painting) is guaranteed to be preserved. This increases the quality and helps with generalization at the cost of training time
    prior_preservation = False  # @param {type:"boolean"}
    # prior_preservation_class_prompt = "a clipart of a bird"
    prior_preservation_class_prompt = "a clipart of a cartoon character"

    num_class_images = 12  # @param {type: "number"}
    # `prior_preservation_weight` determins how strong the class for prior preservation should be
    prior_loss_weight = 1  # @param {type: "number"}

    # If the `prior_preservation_class_folder` is empty, images for the class will be generated with the class prompt. Otherwise, fill this folder with images of items on the same class as your concept (but not images of the concept itself)
    # @param {type:"string"}
    prior_preservation_class_folder = "./class_images"
    class_data_root = prior_preservation_class_folder
    class_prompt = prior_preservation_class_prompt

    # Generate Class Images
    if (prior_preservation):
        class_images_dir = Path(class_data_root)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < num_class_images:
            # sd_pipeline_gen = StableDiffusionPipeline.from_pretrained(
            #     pretrained_model_name_or_path,
            #     revision="fp16",
            #     torch_dtype=torch.float16).to("cuda")

            sd_pipeline_gen = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path).to("cuda")
            sd_pipeline_gen.safety_checker = lambda images, clip_input: (images,
                                                                         False)

            sd_pipeline_gen.enable_attention_slicing()
            sd_pipeline_gen.set_progress_bar_config(disable=True)

            num_new_images = num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset, batch_size=sample_batch_size)

            for example in tqdm(sample_dataloader, desc="Generating class images"):
                images = sd_pipeline_gen(example["prompt"]).images

                for i, image in enumerate(images):
                    image.save(class_images_dir /
                               f"{example['index'][i] + cur_class_images}.jpg")
            pipeline = None
            gc.collect()
            del sd_pipeline_gen
            sd_pipeline_gen = None
            with torch.no_grad():
                torch.cuda.empty_cache()

    # @title Load the Stable Diffusion model
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,
                                                 subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path,
                                        subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path,
                                                subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    # Setting up all training args
    args = Namespace(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        # resolution=vae.sample_size,
        resolution=vae.config.sample_size,

        center_crop=True,
        train_text_encoder=train_text_encoder,
        instance_data_dir=instance_img_dir,
        instance_prompt=instance_prompt,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        save_steps=save_steps,
        train_batch_size=train_batch_size,  # set to 1 if using prior preservation
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        # mixed_precision="fp16",  # set to "fp16" for mixed-precision training.
        mixed_precision="no",  # set to "fp16" for mixed-precision training.
        # set this to True to lower the memory usage.
        gradient_checkpointing=True,
        # use_8bit_adam=True,  # use 8bit optimizer from bitsandbytes
        use_8bit_adam=False,
        seed=seed,
        with_prior_preservation=prior_preservation,
        prior_loss_weight=prior_loss_weight,
        sample_batch_size=sample_batch_size,
        class_daƒta_dir=prior_preservation_class_folder,
        class_prompt=prior_preservation_class_prompt,
        num_class_images=num_class_images,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        output_dir=output_dir,
    )

    def training_function(text_encoder, vae, unet):
        logger = get_logger(__name__)

        set_seed(args.seed)

        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
        # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        vae.requires_grad_(False)
        if not args.train_text_encoder:
            text_encoder.requires_grad_(False)

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if args.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (itertools.chain(unet.parameters(),
                                              text_encoder.parameters())
                              if args.train_text_encoder else unet.parameters())

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
        )

        if (Scheduler_type == "DDIMScheduler"):
            noise_scheduler = DDIMScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = DDPMScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler")

        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir
            if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"]
                         for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # concat class and instance examples for prior preservation
            if args.with_prior_preservation:
                input_ids += [example["class_prompt_ids"]
                              for example in examples]
                pixel_values += [example["class_images"]
                                 for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format).float()

            input_ids = tokenizer.pad(
                {
                    "input_ids": input_ids
                },
                padding="max_length",
                return_tensors="pt",
                max_length=tokenizer.model_max_length).input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn)

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps *
            args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps *
            args.gradient_accumulation_steps,
        )

        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.decoder.to("cpu")
        if not args.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps /
                                     num_update_steps_per_epoch)

        # Train!
        total_batch_size = args.train_batch_size * \
            accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(
            f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(
                        dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps, (bsz, ),
                        device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(
                        latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps,
                                      encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(
                            latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    if args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred,
                                                                   2,
                                                                   dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(),
                                          target.float(),
                                          reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(),
                                                target_prior.float(),
                                                reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(),
                                          target.float(),
                                          reduction="mean")

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = (itertools.chain(
                            unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder else
                            unet.parameters())
                        accelerator.clip_grad_norm_(unet.parameters(),
                                                    args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            pipeline = StableDiffusionPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                unet=accelerator.unwrap_model(unet),
                                text_encoder=accelerator.unwrap_model(
                                    text_encoder),
                            )
                            # pipeline.safety_checker = lambda images, clip_input: (images, False)

                            save_path = os.path.join(args.output_dir,
                                                     f"checkpoint-{global_step}")
                            pipeline.save_pretrained(save_path)

                logs = {"loss": loss.detach().item()}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            # pipeline.safety_checker = lambda images, clip_input: (images, False)
            pipeline.save_pretrained(args.output_dir)

        del pipeline, noise_scheduler
        pipeline, noise_scheduler = None, None
        with torch.no_grad():
            torch.cuda.empty_cache()

    # Run training
    # accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet))
    training_function(text_encoder, vae, unet)

    # free some memory
    for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
        if param.grad is not None:
            del param.grad
        torch.cuda.empty_cache()

    del text_encoder, vae, unet
    text_encoder, vae, unet = None, None, None
    with torch.no_grad():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # python dream_train.py --instance_img_dir 03670_animal_8

    parser = argparse.ArgumentParser(description="Process instance_img_dir.")
    parser.add_argument("--instance_img_dir", type=str,
                        help="The name of the instance image directory")
    args = parser.parse_args()

    # Load options
    dream_opt = option_utils.parse("./dream_train_param.yaml")
    dream_opt = option_utils.dict_to_nonedict(dream_opt)

    model_sv_pa_dir = "./dream_concept/"
    ref_svg_img_collection_dir = "./ref_svg_img_collection/"

    # Process only the specified instance_img_dir
    instance_img_dir = args.instance_img_dir
    print("instance_img_dir: ", instance_img_dir)

    signature = instance_img_dir
    instance_img_dir_whole = ref_svg_img_collection_dir + instance_img_dir + "/"
    output_dir = os.path.join(
        model_sv_pa_dir, signature + "_dream_concept")
    os.makedirs(output_dir, exist_ok=True)

    dream_train(signature=signature, instance_img_dir=instance_img_dir_whole,
                output_dir=output_dir, dream_opt=dream_opt)
