import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src import config

# --- Configuration ---
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 500
SAVE_IMAGE_EPOCHS = 5
SAVE_MODEL_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClassConditionalCellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = config.CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.png"):
                self.images.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Dataset & DataLoader
    preprocess = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = ClassConditionalCellDataset(config.PATCH_CACHE_TRAIN, transform=preprocess)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model & Scheduler
    # We use UNet2DModel which supports class_labels for conditioning out of the box
    model = UNet2DModel(
        sample_size=config.IMG_SIZE,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
        num_class_embeds=len(config.CLASSES), # Enable class conditioning
    ).to(DEVICE)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=(len(train_dataloader) * NUM_EPOCHS),
    )

    # 3. Training Loop
    output_dir = Path("diffusion_checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (clean_images, labels) in enumerate(train_dataloader):
            clean_images = clean_images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with torch.set_grad_enabled(True):
                # Predict the noise residual
                # Pass labels as class_labels for conditioning
                model_output = model(noisy_images, timesteps, class_labels=labels).sample
                loss = F.mse_loss(model_output, noise)
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item())
            global_step += 1
            
        # Optional: Save images or checkpoints
        if (epoch + 1) % SAVE_IMAGE_EPOCHS == 0 or epoch == NUM_EPOCHS - 1:
            evaluate(model, noise_scheduler, epoch, output_dir)
            
        if (epoch + 1) % SAVE_MODEL_EPOCHS == 0 or epoch == NUM_EPOCHS - 1:
            model.save_pretrained(output_dir / f"checkpoint-epoch-{epoch+1}")

def evaluate(model, noise_scheduler, epoch, output_dir):
    # Sample some images from the model to visualize progress
    model.eval()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    
    # Generate 1 image per class
    for i, cls_name in enumerate(config.CLASSES):
        # Conditioning is handled manually in the loop if using DDPMPipeline directly 
        # but DDPMPipeline doesn't natively support class_labels in its __call__.
        # So we manually sample.
        image = sample_conditional(model, noise_scheduler, i)
        image.save(samples_dir / f"epoch_{epoch+1}_{cls_name}.png")
    
    model.train()

def sample_conditional(model, scheduler, class_idx):
    # Standard DDPM sampling loop
    image = torch.randn((1, 3, config.IMG_SIZE, config.IMG_SIZE)).to(DEVICE)
    labels = torch.tensor([class_idx]).to(DEVICE)
    
    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        with torch.no_grad():
            noisy_residual = model(image, t, class_labels=labels).sample
        image = scheduler.step(noisy_residual, t, image).prev_sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).astype("uint8"))
    return image

if __name__ == "__main__":
    train()
