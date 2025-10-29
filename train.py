import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image
import os
import datetime
import numpy as np
import lpips

Image.MAX_IMAGE_PIXELS = None

from torch.utils.tensorboard import SummaryWriter

from unet_architecture import CustomUNet

from geomloss import SamplesLoss

config = {
    "data_dir": "danbooru_dataset",
    "image_size": 256,
    "train_batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "output_dir": "custom_diffusion_model",
    "text_encoder_model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "vae_model": "stabilityai/sdxl-vae",
    "scheduler_source": "runwayml/stable-diffusion-v1-5",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "load_model_path": "custom_diffusion_model_256_v13/unet_final.pth", #update with a previously trained model

    "sample_every_n_steps": 3,
    "sample_prompt": "1boy, red shirt, black pants",
    "sample_output_dir": "sample_images",
    "sampling_steps": 50,
    "guidance_scale": 7.5,
    "epoch_save_count": 1,
    "sample_seed": 800,

    "perceptual_loss_weight": 0,
    "optimal_transport_loss_weight": 0.8,
    "ot_blur": 0.05,
    "ot_p": 2,
    "ot_debias": False,

    "gradient_accumulation_multiplier": 69 #this number plus 1 multiplied by batch size is effective batch size
}


class ImageTextDataset(Dataset):
    def __init__(self, folder_path, tokenizer, image_transforms):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.pairs = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Data directory '{folder_path}' not found.")

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            base_name, _ = os.path.splitext(image_file)
            text_file = base_name + '.txt'
            text_path = os.path.join(folder_path, text_file)
            if os.path.exists(text_path):
                self.pairs.append((os.path.join(folder_path, image_file), text_path))

        if not self.pairs:
            raise ValueError(f"No matching image-text pairs found in '{folder_path}'")
        print(f"Found {len(self.pairs)} image-text pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, text_path = self.pairs[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.image_transforms(image)
        except Exception as e:
            print(f"Warning: Skipping corrupted image: {image_path}. Error: {e}")
            return None

        with open(text_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        inputs = self.tokenizer(
            caption,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "input_ids": input_ids}


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)


print("Loading pre-trained models in FP32...")
tokenizer = CLIPTokenizer.from_pretrained(config["text_encoder_model"])
text_encoder = CLIPTextModel.from_pretrained(config["text_encoder_model"]).to(config["device"])
vae = AutoencoderKL.from_pretrained(config["vae_model"]).to(config["device"])
noise_scheduler = DDIMScheduler.from_pretrained(config["scheduler_source"], subfolder="scheduler")

loss_fn_vgg = None
if config["perceptual_loss_weight"] > 0:
    print("Initializing LPIPS perceptual loss...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(config["device"])
    loss_fn_vgg.requires_grad_(False)
else:
    print("Perceptual loss weight is 0, skipping LPIPS initialization.")

ot_loss_fn = None
if config["optimal_transport_loss_weight"] > 0:
    print(
        f"Initializing SamplesLoss (Sinkhorn) with blur={config['ot_blur']}, p={config['ot_p']}, debias={config['ot_debias']}...")
    ot_loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=config["ot_p"],
        blur=config["ot_blur"],
        debias=config["ot_debias"],
        backend="tensorized",
    ).to(config["device"])
else:
    print("Optimal Transport loss weight is 0, skipping SamplesLoss initialization.")

text_encoder.requires_grad_(False)
vae.requires_grad_(False)

print("Initializing custom U-Net...")
unet = CustomUNet(context_dim=text_encoder.config.hidden_size).to(config["device"])
print(f"U-Net initialized with context_dim = {text_encoder.config.hidden_size}")

if config["load_model_path"]:
    if os.path.exists(config["load_model_path"]):
        print(f"Loading U-Net weights from: {config['load_model_path']}")
        unet.load_state_dict(torch.load(config["load_model_path"], map_location=config["device"]))
    else:
        print(f"Warning: Model path specified but not found: {config['load_model_path']}. Training from scratch.")

print(f"Loading dataset from: {config['data_dir']}")
image_transforms = transforms.Compose(
    [
        transforms.Resize((config["image_size"], config["image_size"]),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
train_dataset = ImageTextDataset(
    folder_path=config["data_dir"],
    tokenizer=tokenizer,
    image_transforms=image_transforms
)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["train_batch_size"], collate_fn=collate_fn)

optimizer = AdamW(unet.parameters(), lr=config["learning_rate"])
if not os.path.exists(config["output_dir"]):
    os.makedirs(config["output_dir"])

log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")

if not os.path.exists(config["sample_output_dir"]):
    os.makedirs(config["sample_output_dir"])


@torch.no_grad()
def generate_sample_image(unet_model, tokenizer_model, text_encoder_model, vae_model, scheduler_model, cfg, global_step,
                          device, writer):
    print(f"\nGenerating sample image at global step {global_step} with prompt: '{cfg['sample_prompt']}'...")
    unet_model.eval()

    text_input = tokenizer_model(
        [cfg['sample_prompt']], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    uncond_input = tokenizer_model(
        [""], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )

    cond_embeddings = text_encoder_model(text_input.input_ids.to(device))[0]
    uncond_embeddings = text_encoder_model(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    scheduler_model.set_timesteps(cfg['sampling_steps'], device=device)

    generator = torch.Generator(device=device).manual_seed(cfg['sample_seed'])
    latents = torch.randn(
        (1, unet_model.conv_in.in_channels, cfg['image_size'] // 8, cfg['image_size'] // 8),
        generator=generator,
        device=device,
    )

    latents = latents * scheduler_model.init_noise_sigma

    for t in tqdm(scheduler_model.timesteps, desc=f"Sampling (Step {global_step})"):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler_model.scale_model_input(latent_model_input, t)
        timestep_tensor = torch.tensor([t], device=device).repeat(latent_model_input.shape[0])
        noise_pred = unet_model(latent_model_input, timestep_tensor, context=text_embeddings)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg['guidance_scale'] * (noise_pred_text - noise_pred_uncond)
        latents = scheduler_model.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image_tensor = vae_model.decode(latents).sample

    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype(np.uint8)

    pil_image = Image.fromarray(image[0])
    sample_image_path = os.path.join(cfg["sample_output_dir"], f"sample_step_{global_step:06d}.png")
    pil_image.save(sample_image_path)
    print(f"Sample image saved to {sample_image_path}")

    writer.add_image("Sample Image", image[0], global_step, dataformats='HWC')

    unet_model.train()


print(f"Starting training on device: {config['device']}")

accumulation_steps = config["gradient_accumulation_multiplier"] + 1
if accumulation_steps > 1:
    print(f"Gradient accumulation enabled. Effective batch size: {config['train_batch_size'] * accumulation_steps}")

global_step = 0
try:
    for epoch in range(config["num_epochs"]):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{config['num_epochs']}")

        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            if batch is None: continue

            clean_images = batch["pixel_values"].to(config["device"])
            input_ids = batch["input_ids"].to(config["device"])

            with torch.no_grad():
                latents = vae.encode(clean_images).latent_dist.sample() * 0.18215
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                      device=config["device"]).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, context=encoder_hidden_states)

            mse_loss = F.mse_loss(noise_pred, noise)
            total_loss = mse_loss

            ot_loss = torch.tensor(0.0, device=config["device"])
            if config["optimal_transport_loss_weight"] > 0 and ot_loss_fn is not None:
                batch_size = noise_pred.shape[0]
                feature_dim = noise_pred.shape[1] * noise_pred.shape[2] * noise_pred.shape[3]
                noise_pred_flat = noise_pred.view(batch_size, feature_dim)
                noise_flat = noise.view(batch_size, feature_dim)
                ot_loss = ot_loss_fn(noise_pred_flat, noise_flat)
                total_loss = total_loss + config["optimal_transport_loss_weight"] * ot_loss

            lpips_loss = torch.tensor(0.0, device=config["device"])
            if config["perceptual_loss_weight"] > 0 and loss_fn_vgg is not None:
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(config["device"])
                sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)
                pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                with torch.no_grad():
                    original_images_reconstructed = vae.decode(latents / 0.18215).sample
                pred_images = vae.decode(pred_original_sample / 0.18215).sample
                lpips_loss = loss_fn_vgg(original_images_reconstructed, pred_images).mean()
                total_loss = total_loss + config["perceptual_loss_weight"] * lpips_loss

            loss = total_loss

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                writer.add_scalar("Loss/total", loss.item() * accumulation_steps, global_step)
                writer.add_scalar("Loss/mse", mse_loss.item(), global_step)
                writer.add_scalar("Loss/lpips", lpips_loss.item(), global_step)
                writer.add_scalar("Loss/ot", ot_loss.item(), global_step)

                progress_bar.set_postfix(
                    step=global_step,
                    total_loss=loss.item() * accumulation_steps,
                    mse=mse_loss.item(),
                    ot=ot_loss.item(),
                    lpips=lpips_loss.item()
                )

                if (global_step % config["sample_every_n_steps"] == 0):
                    generate_sample_image(unet, tokenizer, text_encoder, vae, noise_scheduler, config, global_step,
                                          config["device"], writer)

            progress_bar.update(1)

            if torch.isnan(loss):
                print(f"Loss is NaN at step {global_step}, stopping training.")
                break


        if torch.isnan(loss):
            break

        progress_bar.close()

        if (epoch + 1) % config["epoch_save_count"] == 0:
            torch.save(unet.state_dict(), os.path.join(config["output_dir"], f"unet_epoch_{epoch + 1}.pth"))

except Exception as e:
    print(f"An error occurred during training: {e}")
finally:
    print("Training finished.")
    torch.save(unet.state_dict(), os.path.join(config["output_dir"], "unet_final.pth"))
    writer.close()
    print("TensorBoard writer closed.")