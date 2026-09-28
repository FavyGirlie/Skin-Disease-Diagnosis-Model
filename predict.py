import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
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

# ---------- Input validation thresholds (tune these on real test images) ----------
# The classifier has no "not skin" class, so softmax will always pick one of the
# disease labels, even for a flyer. These gates stop non-skin inputs BEFORE and
# AFTER the network runs.
MIN_IMAGE_SIDE = 64          # pixels; smaller images carry too little detail
MIN_SKIN_RATIO = 0.25        # share of the analysed region that must look like skin
MAX_WHITE_RATIO = 0.35       # share of near-white pixels (paper, documents, flyers)
MAX_DOMINANT_COLOR_RATIO = 0.25  # share taken by ONE exact colour (flat graphics, solid fills)
MAX_FLAT_RATIO = 0.70        # share of perfectly flat pixels (graphics, text on plain paper)
MIN_CONFIDENCE = 50.0        # percent; below this the result is not shown at all


# ==================== RESULT TYPE ====================
@dataclass
class Prediction:
    """Outcome of one screening.

    status is one of:
      "ok"           the image passed all checks and label/confidence are valid
      "not_skin"     the image does not look like human skin, no diagnosis is made
      "inconclusive" it looks like skin but the model is not confident enough
    """

    status: str
    label: Optional[str]
    confidence: float
    message: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, float] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "ok"


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
# The crop step is kept separate so the skin check can inspect exactly the
# region the network will see.
crop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225],   # ImageNet stds
    ),
])

# Kept for backward compatibility with any code that imports `transform`.
transform = transforms.Compose([crop_transform, tensor_transform])


# ==================== INPUT VALIDATION ====================
def analyze_image_content(image: Image.Image) -> Dict[str, float]:
    """Measure simple, model-free statistics of the region the network will see.

    skin_ratio  share of pixels whose colour lies in a skin chroma range
                (YCbCr based, so it covers light and dark skin tones, which
                differ mostly in brightness rather than in chroma)
    white_ratio share of near-white pixels
    dominant_color_ratio
                share of pixels that have exactly the same RGB value as the most
                common colour (high for graphics and solid backgrounds, tiny for
                camera photos, even smooth ones)
    flat_ratio  share of pixels identical to their right and lower neighbours
    """
    view = crop_transform(image.convert("RGB"))
    rgb = np.asarray(view, dtype=np.uint8)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)

    ycc = np.asarray(view.convert("YCbCr"), dtype=np.float32)
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = (
        (y > 30)
        & (cb >= 77) & (cb <= 127)
        & (cr >= 133) & (cr <= 180)
        & (r > g) & (r > b)
    )

    white = (r >= 240) & (g >= 240) & (b >= 240)

    same_right = (rgb[:, :-1, :] == rgb[:, 1:, :]).all(axis=2)[:-1, :]
    same_down = (rgb[:-1, :, :] == rgb[1:, :, :]).all(axis=2)[:, :-1]
    flat = same_right & same_down

    packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    _, counts = np.unique(packed, return_counts=True)
    dominant = counts.max() / packed.size

    return {
        "skin_ratio": float(skin.mean()),
        "white_ratio": float(white.mean()),
        "dominant_color_ratio": float(dominant),
        "flat_ratio": float(flat.mean()),
    }


def check_is_skin_image(image: Image.Image):
    """Return (is_skin, reason, stats). reason is empty when the image passes."""
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        return (
            False,
            f"The image is too small ({width} x {height} px).",
            {"width": width, "height": height},
        )

    stats = analyze_image_content(image)

    if stats["skin_ratio"] < MIN_SKIN_RATIO:
        return (
            False,
            "Too little of the image looks like human skin.",
            stats,
        )
    if stats["white_ratio"] > MAX_WHITE_RATIO:
        return (
            False,
            "The image is mostly white, like a document, flyer or screenshot.",
            stats,
        )
    if (
        stats["dominant_color_ratio"] > MAX_DOMINANT_COLOR_RATIO
        or stats["flat_ratio"] > MAX_FLAT_RATIO
    ):
        return (
            False,
            "The image looks like a graphic or design, not a photograph of skin.",
            stats,
        )
    return True, "", stats


# ==================== PREDICTION ====================
NOT_SKIN_MESSAGE = (
    "This does not look like a photo of human skin, so no screening was done. "
    "Please upload a clear, well-lit, close-up photo of the affected skin area. "
    "Documents, flyers, screenshots, objects and scenes cannot be screened."
)

INCONCLUSIVE_MESSAGE = (
    "The image looks like skin, but the model could not reach a reliable result. "
    "Try a sharper, better-lit, closer photo of the affected area, "
    "or consult a dermatologist directly."
)


def predict_image(
    image: Union[str, Image.Image],
    model: nn.Module,
    check_skin: bool = True,
) -> Prediction:
    """Screen a file path or PIL image and return a Prediction.

    Non-skin images are rejected before the network runs. Skin images with a
    low top probability are reported as inconclusive instead of being given a
    label.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image).convert("RGB")

    checks: Dict[str, float] = {}
    if check_skin:
        is_skin, reason, checks = check_is_skin_image(image)
        if not is_skin:
            if os.environ.get("SKIN_APP_DEBUG"):
                print(f"Rejected as not skin: {reason} {checks}", flush=True)
            return Prediction(
                status="not_skin",
                label=None,
                confidence=0.0,
                message=f"{NOT_SKIN_MESSAGE}\n\nReason: {reason}",
                checks=checks,
            )

    input_tensor = tensor_transform(crop_transform(image)).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    probs_by_class = {
        name: prob * 100 for name, prob in zip(CLASS_NAMES, probabilities.tolist())
    }
    confidence_pct = confidence.item() * 100

    if os.environ.get("SKIN_APP_DEBUG"):
        # Max logit and energy are useful for calibrating further rejection
        # thresholds on your own non-skin test images.
        logits = outputs[0]
        print(f"   max logit: {logits.max().item():.3f}", flush=True)
        print(f"   energy:    {-torch.logsumexp(logits, dim=0).item():.3f}", flush=True)
        for name, prob in probs_by_class.items():
            print(f"   {name}: {prob:.2f}%", flush=True)

    if confidence_pct < MIN_CONFIDENCE:
        return Prediction(
            status="inconclusive",
            label=None,
            confidence=confidence_pct,
            message=INCONCLUSIVE_MESSAGE,
            probabilities=probs_by_class,
            checks=checks,
        )

    return Prediction(
        status="ok",
        label=CLASS_NAMES[predicted_idx.item()],
        confidence=confidence_pct,
        message="",
        probabilities=probs_by_class,
        checks=checks,
    )


# ==================== LOCAL TEST ====================
if __name__ == "__main__":
    net = load_model(get_model_path())
    result = predict_image("test_image.jpg", net)
    if result.accepted:
        print(f"Diagnosis: {result.label}")
        print(f"Confidence: {result.confidence:.2f}%")
    else:
        print(f"Status: {result.status}")
        print(result.message)
