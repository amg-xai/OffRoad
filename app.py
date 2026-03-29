import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# ============================================================================
# Config
# ============================================================================

n_classes = 11

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

color_palette = np.array([
    [0, 0, 0],
    [34, 139, 34],
    [0, 255, 0],
    [210, 180, 140],
    [139, 90, 43],
    [128, 128, 0],
    [255, 215, 0],
    [139, 69, 19],
    [128, 128, 128],
    [160, 82, 45],
    [135, 206, 235],
], dtype=np.uint8)

# ============================================================================
# Model
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Load Models (once at startup)
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w = int(((960 / 2) // 14) * 14)
h = int(((540 / 2) // 14) * 14)

print("Loading DINOv2 backbone...")
backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
backbone_model.eval()
backbone_model.to(device)

print("Loading segmentation head...")
classifier = SegmentationHeadConvNeXt(
    in_channels=384,
    out_channels=n_classes,
    tokenW=w // 14,
    tokenH=h // 14
)
classifier.load_state_dict(torch.load("segmentation_head.pth", map_location=device))
classifier.eval()
classifier.to(device)
print("Models loaded!")

# ============================================================================
# Transform
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# Inference
# ============================================================================

def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone_model.forward_features(img_tensor)["x_norm_patchtokens"]
        logits = classifier(features)
        outputs = F.interpolate(logits, size=img_tensor.shape[2:], mode="bilinear", align_corners=False)

    pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    color_mask = mask_to_color(pred_mask)

    return Image.fromarray(color_mask)


# ============================================================================
# Legend
# ============================================================================

legend_html = "<div style='display:flex; flex-wrap:wrap; gap:8px; padding:10px'>"
for i, name in enumerate(class_names):
    r, g, b = color_palette[i]
    legend_html += f"<div style='display:flex; align-items:center; gap:5px'><div style='width:20px; height:20px; background:rgb({r},{g},{b}); border:1px solid #ccc'></div><span>{name}</span></div>"
legend_html += "</div>"

# ============================================================================
# Gradio UI
# ============================================================================

with gr.Blocks(title="OffRoad Segmentation") as demo:
    gr.Markdown("# 🌿 OffRoad Scene Segmentation")
    gr.Markdown("Upload an offroad image to get a segmentation mask showing terrain classes.")

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        output_image = gr.Image(type="pil", label="Segmentation Mask")

    gr.Button("Segment", variant="primary").click(
        fn=predict,
        inputs=input_image,
        outputs=output_image
    )

    gr.Markdown("### Class Legend")
    gr.HTML(legend_html)

demo.launch()