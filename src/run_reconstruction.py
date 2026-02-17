from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. LOAD IMAGE
image_path = "campanile.jpeg" 
if os.path.exists(image_path):
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded {image_path}")
else:
    raise FileNotFoundError(f"Could not find {image_path}.")

# 2. DEPTH ESTIMATION
print("Running Depth Anything V2...")
image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

inputs = image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth_array = predicted_depth.squeeze().cpu().numpy()
height, width = depth_array.shape 

# 3. SEGMENTATION (SAM 2)
print("Running SAM 2 Isolation...")
checkpoint = "sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
device = "mps" if torch.backends.mps.is_available() else "cpu"
sam2_model = build_sam2(model_cfg, checkpoint, device=device) 
predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

# UPDATED: Clicking lower (Center of image) to ensure we hit the tower
input_point = np.array([[width // 2, height // 2]]) 
input_label = np.array([1]) 

masks, _, _ = predictor.predict(input_point, input_label, multimask_output=False)

# --- DEFINING building_mask HERE ---
building_mask = np.array(masks[0]).astype(bool).squeeze()

# --- DEBUGGING PREVIEW (Now building_mask is defined!) ---
mask_preview = Image.fromarray((building_mask * 255).astype(np.uint8))
mask_preview.save("sam2_check.jpg")
print("Saved 'sam2_check.jpg'. Open this file to see what SAM 2 selected!")

# --- STEP 4: PROPORTIONAL 3D PROJECTION ---
print("Creating 3D Point Cloud...")
stride = 1 # Use every pixel
sub_mask = building_mask[::stride, ::stride]
sub_depth = depth_array[::stride, ::stride]
sub_image = np.array(image)[::stride, ::stride]

h_sub, w_sub = sub_depth.shape
u, v = np.meshgrid(np.arange(w_sub), np.arange(h_sub))

# Filter
u_f, v_f, z_raw = u[sub_mask], v[sub_mask], sub_depth[sub_mask]

# Instead of dividing by 1000, we want Z to be in the same range as u and v
# 0.0 to 1.0 depth * 5000 (roughly the width of your image)
z_scaled = z_raw * 5000.0 

# Pinhole Math (Simplified for visual matching)
focal_length = 2856 
x = (u_f - (width / 2)) * z_scaled / focal_length
y = (v_f - (height / 2)) * z_scaled / focal_length
z = z_scaled

points = np.stack((x, y, z), axis=-1)
colors = sub_image[sub_mask] / 255.0

# 5. SAVE
def save_colored_ply(filename, pts, cols):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(len(pts)):
            p, c = pts[i], (cols[i] * 255).astype(int)
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

save_colored_ply("campanile_final.ply", points, colors)
print(f"Success! Points saved: {len(points)}")