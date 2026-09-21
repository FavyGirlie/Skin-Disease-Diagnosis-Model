import os
from typing import Union

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageOps
from huggingface_hub import hf_hub_download

# ==================== CONFIG ====================
# CPU is fine for lightweight web inference; falls back automatically.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.pt is downloaded from the Hugging Face Hub and cached locally.
HF_REPO_ID = "De-FavouredOne/Skin-Disease-Diagnosis-Model"
HF_FILENAME = "model.pt"

# IMPORTANT: the order must be identical to the label indices used in training.
# If you trained with torchvision ImageFolder, indices follow alphabetical order
# of the class folders. The number of names must equal the number of outputs of
# the final layer in your checkpoint (load_model checks this and tells you).
CLASS_NAMES = [
    "Eczema",
    "Fungal",
    "Others",
    "Scabies",
    "Dermatitis",
]

# Real ResNet50 weights are roughly 90 MB or more. Anything far smaller is
# almost certainly a Git LFS pointer or a broken download.
MIN_EXPECTED_BYTES = 5 * 1024 * 1024


# ==================== MODEL ARCHITECTURE ====================
# The checkpoint keys are prefixed "backbone.*" and match a ResNet50
# (bottleneck blocks 3-4-6-3) with a custom classifier head.
# The architecture must match exactly or the trained weights will not load.
class ResNet50Model(nn.Module):
    def __init__(self, num_classes: int):
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


# ==================== DOWNLOAD ====================
def get_model_path() -> str:
    """Download (or reuse the cached) model file and return its local path.

    Raises an exception with a readable message on any failure, so the caller
    can display the real reason instead of a silent None.
    """
    token = os.environ.get("HF_TOKEN")  # only needed if the repo is private

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=token)

    # A stale or partially cleaned cache can leave a dangling symlink.
    # In that case, force a clean re-download once.
    if not os.path.isfile(path):
        print(f"Cached path is missing or broken, re-downloading: {path}", flush=True)
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=token,
            force_download=True,
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"hf_hub_download returned a path that does not exist: {path}"
        )
    return path


# ==================== LOADING ====================
def _check_weights_file(path: str) -> None:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(64)

    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{HF_FILENAME} is a Git LFS pointer file ({size} bytes), not real weights. "
            "Re-upload the model to the Hub."
        )
    if size < MIN_EXPECTED_BYTES:
        raise RuntimeError(
            f"{HF_FILENAME} is only {size} bytes, which is too small for ResNet50 weights. "
            "The upload or download is probably incomplete."
        )
    print(f"Model file OK: {path} ({size / (1024 * 1024):.2f} MB)", flush=True)


def _read_checkpoint(path: str):
    # Try the safe loader first (works for plain state dicts).
    # Fall back to full unpickling for whole-model checkpoints.
    # Only do the fallback for files you created yourself.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as safe_err:
        print(f"weights_only=True failed ({type(safe_err).__name__}), retrying with weights_only=False", flush=True)
        return torch.load(path, map_location=device, weights_only=False)


def _extract_state_dict(checkpoint: dict) -> dict:
    # Unwrap common checkpoint layouts.
    for key in ("model_state_dict", "state_dict", "model"):
        inner = checkpoint.get(key)
        if isinstance(inner, dict):
            checkpoint = inner
            break
    # Remove the "module." prefix added by DataParallel, if present.
    return {k.removeprefix("module."): v for k, v in checkpoint.items()}


def load_model(model_path: str) -> nn.Module:
    """Load the model and return it in eval mode.

    Raises an exception with a specific message on any failure. Nothing is
    swallowed, so a failed load can never be cached as a silent None.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path} (cwd: {os.getcwd()})"
        )

    _check_weights_file(model_path)
    checkpoint = _read_checkpoint(model_path)

    # Case 1: the file contains a whole pickled model.
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        checkpoint.to(device)
        print("Loaded a full model object.", flush=True)
        return checkpoint

    # Case 2: the file contains weights only.
    if isinstance(checkpoint, dict):
        state = _extract_state_dict(checkpoint)

        fc_key = "backbone.fc.5.weight"
        if fc_key not in state:
            sample = list(state.keys())[:5]
            raise KeyError(
                f"Expected key '{fc_key}' not found. First keys in checkpoint: {sample}. "
                "The saved architecture does not match ResNet50Model."
            )

        ckpt_classes = state[fc_key].shape[0]
        if ckpt_classes != len(CLASS_NAMES):
            raise ValueError(
                f"Checkpoint has {ckpt_classes} output classes but CLASS_NAMES has "
                f"{len(CLASS_NAMES)}. Fix CLASS_NAMES in predict.py so it matches the "
                "classes and the order used in training."
            )

        model = ResNet50Model(num_classes=ckpt_classes)
        # strict=True on purpose: a mismatch must fail loudly instead of
        # leaving a partly untrained model.
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(device)
        print(f"Loaded weights into ResNet50 ({ckpt_classes} classes).", flush=True)
        return model

    raise ValueError(f"Unrecognized checkpoint format: {type(checkpoint)}")


# ==================== PREPROCESSING ====================
# This must match the validation/test transforms used in training.
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225],   # ImageNet stds
    ),
])


# ==================== PREDICTION ====================
def predict_image(image: Union[str, Image.Image], model: nn.Module):
    """Return (label, confidence_percent) for a file path or a PIL image."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    if os.environ.get("SKIN_APP_DEBUG"):
        for name, prob in zip(CLASS_NAMES, probabilities.tolist()):
            print(f"   {name}: {prob * 100:.2f}%", flush=True)

    return CLASS_NAMES[predicted_idx.item()], confidence.item() * 100


# ==================== LOCAL TEST ====================
if __name__ == "__main__":
    net = load_model(get_model_path())
    label, conf = predict_image("test_image.jpg", net)
    print(f"Diagnosis: {label}")
    print(f"Confidence: {conf:.2f}%")
