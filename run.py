import torch
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

from unet_architecture import CustomUNet

config = {
    "prompt": input("prompt: "),
    "batch_size": int(input("Number of images to generate (batch size): ")),
    "unet_path": "custom_diffusion_model_256_v14/unet_final.pth",
    "output_path": "generated_image_batch.png",
    "steps": 50,
    "guidance_scale": 7.5,
    "height": 256,
    "width": 256,
    "text_encoder_model": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "vae_model": "stabilityai/sdxl-vae",
    "scheduler_source": "runwayml/stable-diffusion-v1-5",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main():
    print(f"Using device: {config['device']}")
    device = config['device']

    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(config['text_encoder_model'])
    text_encoder = CLIPTextModel.from_pretrained(config['text_encoder_model']).to(device)
    vae = AutoencoderKL.from_pretrained(config['vae_model']).to(device)
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(config['scheduler_source'], subfolder="scheduler")

    print(f"Loading trained U-Net weights from: {config['unet_path']}")
    unet = CustomUNet(context_dim=text_encoder.config.hidden_size).to(device)
    try:
        unet.load_state_dict(torch.load(config['unet_path'], map_location=device))
    except FileNotFoundError:
        print(f"Error: U-Net weights file not found at '{config['unet_path']}'.")
        return

    unet.eval()

    print("Preparing text embeddings...")
    batch_size = config['batch_size']
    text_input = tokenizer(
        [config['prompt']], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        cond_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    cond_embeddings = cond_embeddings.repeat(batch_size, 1, 1)
    uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)

    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    print(f"Starting diffusion process for batch of {batch_size}...")
    scheduler.set_timesteps(config['steps'])

    latents = torch.randn(
        (batch_size, unet.conv_in.in_channels, config['height'] // 8, config['width'] // 8),
        device=device,
    )
    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        timestep = torch.tensor([t], device=device).repeat(latent_model_input.shape[0])

        with torch.no_grad():
            noise_pred = unet(latent_model_input, timestep, context=text_embeddings)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config['guidance_scale'] * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    print("Decoding image batch with VAE...")
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    image = (image_tensor / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype(np.uint8)

    pil_images = [Image.fromarray(img) for img in image]

    total_width = config['width'] * batch_size
    height = config['height']
    composite_image = Image.new('RGB', (total_width, height))

    x_offset = 0
    for img in pil_images:
        composite_image.paste(img, (x_offset, 0))
        x_offset += img.width

    composite_image.save(config['output_path'])
    print(f"Image batch saved successfully to {config['output_path']}")


if __name__ == "__main__":
    main()