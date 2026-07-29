import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from huggingface_hub import hf_hub_download

# 1. Set Device (CPU is ideal for lightweight web inference)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Define Model Path and Target Classes
# Downloads model.pt from Hugging Face Hub (cached locally after first run)
HF_REPO_ID = "De-FavouredOne/Skin-Disease-Diagnosis"  
HF_FILENAME = "model.pt"

DOWNLOAD_ERROR = None  # app.py can check this if MODEL_PATH is None

try:
    MODEL_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
except Exception as e:
    MODEL_PATH = None
    DOWNLOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"❌ Error downloading model from Hugging Face: {e}", flush=True)

# PASSION Dataset Classes
# Update this list with your actual PASSION dataset class names
CLASS_NAMES = [
    'Eczema',
    'Fungal',
    'Others',
    'Scabies'
]

# 3. Model Architecture
# The checkpoint's state dict keys are prefixed "backbone.*" and match a
# ResNet50 (bottleneck blocks, 3-4-6-3) with a custom classifier head —
# NOT plain ResNet18. Architecture must match exactly (verified via a
# strict=True load against models/model.pt: all 327 keys matched) or
# load_state_dict silently drops the trained weights.
class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

# 4. Load the Model
LAST_LOAD_ERROR = None  # app.py reads this to show the real failure reason

def load_model(model_path):
    global LAST_LOAD_ERROR
    LAST_LOAD_ERROR = None
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print(f"📁 Current directory: {os.getcwd()}")
            print(f"📁 Files in current directory: {os.listdir('.')}")
            return None

        file_size = os.path.getsize(model_path)
        print(f"📦 Loading model from: {model_path} ({file_size / (1024*1024):.2f} MB)")

        # Load the checkpoint dictionary or model object onto the specified device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # If the loaded object is already a full model:
        if isinstance(checkpoint, torch.nn.Module):
            checkpoint.eval()
            checkpoint.to(device)
            print("✓ Model (full module) loaded successfully!")
            return checkpoint

        # If it's a state_dict or OrderedDict (weights only)
        elif isinstance(checkpoint, (dict, torch.nn.modules.container.OrderedDict)):
            num_classes = len(CLASS_NAMES)
            print(f"ℹ️ Loading weights into ResNet50 architecture ({num_classes} classes)...")
            model = ResNet50Model(num_classes=num_classes)

            # strict=True on purpose: a mismatch here means the trained weights
            # would otherwise silently fail to load, leaving an untrained model.
            model.load_state_dict(checkpoint, strict=True)
            model.eval()
            model.to(device)
            print(f"✓ Model weights loaded successfully into ResNet50 (all keys matched)!")
            return model
        else:
            print(f"ℹ️ Attempting to use loaded object directly...")
            if hasattr(checkpoint, 'eval'):
                checkpoint.eval()
                checkpoint.to(device)
                print("✓ Model loaded successfully!")
                return checkpoint
            raise ValueError(f"Unrecognized model format: {type(checkpoint)}")

    except Exception as e:
        LAST_LOAD_ERROR = f"{type(e).__name__}: {e}"
        print(f"❌ Error loading model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# 5. Define Standard Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet standard means
        std=[0.229, 0.224, 0.225]    # ImageNet standard stds
    )
])

# 6. Prediction Function
def predict_image(image_path, model):
    # Load image and convert to RGB
    image = Image.open(image_path).convert('RGB')

    # Preprocess image and add batch dimension [1, 3, 224, 224]
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Disable gradient calculations during inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    # 🔍 DEBUG: print raw probabilities for each class to see if the model
    # is actually reacting to the image, or just always favoring one class
    print("=" * 50)
    print(f"🔍 DEBUG - Raw probabilities per class:")
    for cls_name, prob in zip(CLASS_NAMES, probabilities.tolist()):
        print(f"   {cls_name}: {prob * 100:.2f}%")
    print("=" * 50)

    prediction = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    return prediction, confidence_pct

# --- Example Execution ---
if __name__ == "__main__":
    model = load_model(MODEL_PATH)

    if model:
        image_path = "test_image.jpg"  # Path to a test image
        label, conf = predict_image(image_path, model)
        print(f"Diagnosis: {label}")
        print(f"Confidence: {conf:.2f}%")