import torch
from diffusers import UNet2DConditionModel
import json

### After training finish, you can convert checkpoint .pth into .safetensors using this .py to make 
### easy integration of Stable-Diffusion pipeline.

with open("./configs/unet/config_mri.json", "r") as f:
    config = json.load(f)
# Path to your .pth file
pth_path = "../output/checkpoint-step-#/unet_ema_0.999_weights.pth"
# pth_path = "/home/work/dohun2/MRI-DM/output_clip/checkpoint-step-140000/unet_weights.pth"
# Load the state dictionary from the .pth file
state_dict = torch.load(pth_path, map_location="cpu")

# Optionally: create a Hugging Face model instance if the model architecture is known
model = UNet2DConditionModel.from_config(config)  # Replace with the appropriate model
model.load_state_dict(state_dict)

# Save the model in Hugging Face's format (.bin)
model.save_pretrained("./MRI_checkpoint/unet")