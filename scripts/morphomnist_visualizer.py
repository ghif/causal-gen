#!/usr/bin/env python3
"""Interactive MorphoMNIST causal visualizer.

Launch:

    python scripts/morphomnist_visualizer.py \
        --checkpoint gs://medical-airnd/causal-gen/checkpoints/t_i_d/cf_torch-gpu-g4_17-07-2026

The app exposes two workflows:

- generate images directly from causal-factor sliders
- upload a seed digit and render a counterfactual edit from the same model
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

cache_root = Path(tempfile.gettempdir()) / "causal-gen-cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import huggingface_hub as _huggingface_hub

    if not hasattr(_huggingface_hub, "HfFolder"):
        class _HfFolder:
            @staticmethod
            def get_token() -> str | None:
                return None

            @staticmethod
            def save_token(token: str) -> None:
                return None

            @staticmethod
            def delete_token() -> None:
                return None

        _huggingface_hub.HfFolder = _HfFolder  # type: ignore[attr-defined]
except Exception:
    pass

import gradio as gr


def patch_gradio_runtime_compatibility() -> None:
    """Bridge Gradio 4 APIs to newer Pydantic and Starlette releases."""
    try:
        from gradio_client import utils as client_utils
    except ImportError:
        client_utils = None

    if client_utils is not None:
        converter = getattr(client_utils, "_json_schema_to_python_type", None)
        if converter is not None and not getattr(converter, "_causal_gen_compatible", False):
            def compatible_converter(schema, defs):
                # JSON Schema permits boolean schemas; Gradio 4.44 assumes a mapping.
                if isinstance(schema, bool):
                    return "Any"
                return converter(schema, defs)

            compatible_converter._causal_gen_compatible = True
            client_utils._json_schema_to_python_type = compatible_converter

    try:
        from starlette.templating import Jinja2Templates
    except ImportError:
        return

    template_response = Jinja2Templates.TemplateResponse
    if getattr(template_response, "_causal_gen_compatible", False):
        return

    def compatible_template_response(self, *args, **kwargs):
        # Gradio 4 passes (name, context); Starlette 1.x expects (request, name, context).
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name, context = args[:2]
            request = context.get("request")
            return template_response(
                self, request, name, context, *args[2:], **kwargs
            )
        return template_response(self, *args, **kwargs)

    compatible_template_response._causal_gen_compatible = True
    Jinja2Templates.TemplateResponse = compatible_template_response


patch_gradio_runtime_compatibility()

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
PGM_DIR = SRC_DIR / "pgm"
for path in (str(SRC_DIR), str(PGM_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from hps import Hparams  # noqa: E402
from pgm.dscm import DSCM  # noqa: E402
from pgm.flow_pgm import MorphoMNISTPGM  # noqa: E402
from utils import EMA, open_file, path_exists, select_device, seed_all  # noqa: E402
from vae import HVAE  # noqa: E402


MORPHO_MIN_MAX = {
    "thickness": (0.87598526, 6.255515),
    "intensity": (66.601204, 254.90317),
}

APP_CSS = """
#generated-preview img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    image-rendering: pixelated;
}
"""


def normalize_value(value: float, key: str) -> float:
    min_v, max_v = MORPHO_MIN_MAX[key]
    if max_v <= min_v:
        return 0.0
    normalized = ((value - min_v) / (max_v - min_v)) * 2.0 - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def denormalize_value(value: float, key: str) -> float:
    min_v, max_v = MORPHO_MIN_MAX[key]
    scaled = ((float(value) + 1.0) / 2.0) * (max_v - min_v) + min_v
    return float(np.clip(scaled, min_v, max_v))


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x[0]
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).numpy()
    return Image.fromarray(x, mode="L")


def preprocess_seed_image(image: Image.Image, input_res: int) -> torch.Tensor:
    image = image.convert("L")
    if image.size == (input_res, input_res):
        tensor = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0)
        return (tensor - 127.5) / 127.5

    if image.size != (28, 28):
        image = image.resize((28, 28), Image.BILINEAR)
    image = ImageOps.expand(image, border=2, fill=0)
    tensor = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0)
    return (tensor - 127.5) / 127.5


def one_hot_digit(digit: int, device: torch.device) -> torch.Tensor:
    return F.one_hot(torch.tensor([int(digit)], device=device), num_classes=10).float()


def build_parent_dict(
    digit: int, thickness: float, intensity: float, device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "thickness": torch.tensor(
            [[normalize_value(thickness, "thickness")]], device=device, dtype=torch.float32
        ),
        "intensity": torch.tensor(
            [[normalize_value(intensity, "intensity")]], device=device, dtype=torch.float32
        ),
        "digit": one_hot_digit(digit, device),
    }


def summarize_parents(pa: Dict[str, torch.Tensor]) -> Dict[str, float]:
    digit = int(pa["digit"].detach().cpu().argmax(dim=-1).item())
    thickness = denormalize_value(float(pa["thickness"].detach().cpu().item()), "thickness")
    intensity = denormalize_value(float(pa["intensity"].detach().cpu().item()), "intensity")
    return {
        "digit": digit,
        "thickness": round(thickness, 3),
        "intensity": round(intensity, 3),
    }


def vae_preprocess(args: Hparams, pa: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    concat_pa = torch.cat(
        [pa[k] if len(pa[k].shape) > 1 else pa[k][..., None] for k in args.parents_x],
        dim=1,
    )
    concat_pa = concat_pa[..., None, None].repeat(1, 1, args.input_res, args.input_res)
    return concat_pa.to(device).float()


def infer_dataset(hparams: Hparams, fallback: str = "morphomnist") -> str:
    candidates = []
    if hasattr(hparams, "dataset") and getattr(hparams, "dataset"):
        candidates.append(str(getattr(hparams, "dataset")).lower())
    if hasattr(hparams, "hps") and getattr(hparams, "hps"):
        candidates.append(str(getattr(hparams, "hps")).lower())

    for candidate in candidates:
        if "morphomnist" in candidate:
            return "morphomnist"
        if "cmnist" in candidate:
            return "cmnist"
        if "mimic" in candidate:
            return "mimic"
        if "ukbb" in candidate:
            return "ukbb"
    return fallback


def ensure_dataset_defaults(args: Hparams) -> None:
    dataset = infer_dataset(args)
    args.dataset = dataset
    if not hasattr(args, "hps") or not getattr(args, "hps"):
        args.hps = dataset

    if "morphomnist" in dataset:
        args.input_channels = getattr(args, "input_channels", 1)
        args.input_res = getattr(args, "input_res", 32)
        args.pad = getattr(args, "pad", 4)
        args.parents_x = getattr(args, "parents_x", ["thickness", "intensity", "digit"])
        args.concat_pa = getattr(args, "concat_pa", False)
        args.context_norm = getattr(args, "context_norm", "[-1,1]")
        args.context_dim = getattr(args, "context_dim", 12)
        args.enc_arch = getattr(
            args,
            "enc_arch",
            "32b3d2,16b3d2,8b3d2,4b3d4,1b4",
        )
        args.dec_arch = getattr(
            args,
            "dec_arch",
            "1b4,4b4,8b4,16b4,32b4",
        )
        args.widths = getattr(args, "widths", [16, 32, 64, 128, 256])
    elif "cmnist" in dataset:
        args.input_channels = getattr(args, "input_channels", 3)
        args.input_res = getattr(args, "input_res", 32)
        args.pad = getattr(args, "pad", 4)
        args.parents_x = getattr(args, "parents_x", ["digit", "colour"])
        args.concat_pa = getattr(args, "concat_pa", False)
    elif "ukbb" in dataset:
        args.input_channels = getattr(args, "input_channels", 1)
        args.input_res = getattr(args, "input_res", 192)
        args.pad = getattr(args, "pad", 9)
    elif "mimic" in dataset:
        args.input_channels = getattr(args, "input_channels", 1)
        args.input_res = getattr(args, "input_res", 192)
        args.pad = getattr(args, "pad", 9)


def ensure_vae_defaults(args: Hparams) -> None:
    args.bottleneck = getattr(args, "bottleneck", 4)
    args.z_max_res = getattr(args, "z_max_res", getattr(args, "input_res", 32))
    args.bias_max_res = getattr(args, "bias_max_res", getattr(args, "input_res", 32))
    args.hidden_dim = getattr(args, "hidden_dim", 128)
    args.x_like = getattr(args, "x_like", "diag_dgauss")
    args.std_init = getattr(args, "std_init", 0.0)
    args.cond_prior = getattr(args, "cond_prior", False)
    args.q_correction = getattr(args, "q_correction", False)
    args.kl_free_bits = getattr(args, "kl_free_bits", getattr(args, "free_bits", 0.0))
    args.free_bits = getattr(args, "free_bits", args.kl_free_bits)


def ensure_setup(args: Hparams, default_setup: str) -> None:
    if not hasattr(args, "setup") or not getattr(args, "setup"):
        args.setup = default_setup


def load_checkpoint(path: str) -> Dict:
    resolved_path = resolve_checkpoint_path(path)
    if not resolved_path:
        raise FileNotFoundError(
            f"Could not find a serialized checkpoint file under: {path}"
        )
    with open_file(resolved_path, "rb") as f:
        return torch.load(f, map_location="cpu")


def _entry_timestamp(entry) -> float:
    mtime = entry.get("mtime", 0.0) if isinstance(entry, dict) else 0.0
    if isinstance(mtime, datetime):
        return mtime.timestamp()
    if hasattr(mtime, "timestamp"):
        try:
            return float(mtime.timestamp())
        except Exception:
            return 0.0
    try:
        return float(mtime)
    except Exception:
        return 0.0


def _sort_key(entry) -> Tuple[int, float, str]:
    name = entry.get("name") if isinstance(entry, dict) else os.path.basename(str(entry))
    size = entry.get("size", 0) if isinstance(entry, dict) else 0
    return (0 if name.endswith(".pt") else 1, -_entry_timestamp(entry), f"{size:020d}:{name}")


def resolve_checkpoint_path(path: str) -> str:
    """Resolve a checkpoint file from either a file path or a checkpoint directory."""
    if path.endswith(".pt") or path.endswith(".pth"):
        return path if path_exists(path) else ""

    # Direct file with no extension or a directory may be passed in. Prefer
    # canonical checkpoint filenames before falling back to the newest *.pt file.
    candidates = [
        os.path.join(path, "checkpoint.pt"),
        os.path.join(path, "model.pt"),
        os.path.join(path, "ema_checkpoint.pt"),
    ]
    for candidate in candidates:
        if path_exists(candidate):
            return candidate

    # Search for checkpoint files, preferring the newest stored checkpoint.
    try:
        if path.startswith("gs://"):
            import fsspec

            fs = fsspec.filesystem("gcs")
            try:
                dir_info = fs.info(path)
            except Exception:
                dir_info = {"type": "directory"}
            if dir_info.get("type") == "file" and dir_info.get("size", 0) > 0:
                return path
            try:
                entries = fs.find(path, detail=True)
            except Exception:
                entries = fs.listdir(path, detail=True)
            files = []
            iterable = entries.values() if isinstance(entries, dict) else entries
            for entry in iterable:
                name = entry.get("name", "")
                entry_type = entry.get("type", "")
                if entry_type == "directory":
                    continue
                if name.endswith(".pt") or name.endswith(".pth"):
                    files.append(entry)
            if files:
                files.sort(key=_sort_key)
                name = files[0]["name"]
                return name if name.startswith("gs://") else f"gs://{name}"
        else:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
            if os.path.isdir(path):
                files = []
                for pattern in ("*.pt", "*.pth"):
                    files.extend(glob.glob(os.path.join(path, "**", pattern), recursive=True))
                if files:
                    files.sort(
                        key=lambda p: (
                            0 if p.endswith(".pt") else 1,
                            -os.path.getmtime(p),
                            p,
                        )
                    )
                    return files[0]
    except Exception as exc:
        raise RuntimeError(f"Failed to resolve checkpoint path '{path}': {exc}") from exc

    return ""


def build_pgm(args: Hparams) -> torch.nn.Module:
    if "morphomnist" in args.dataset:
        return MorphoMNISTPGM(args)
    raise NotImplementedError(f"Unsupported dataset: {args.dataset}")


def load_pgm_from_checkpoint(path: str, device: torch.device, setup: str) -> torch.nn.Module:
    ckpt = load_checkpoint(path)
    args = Hparams()
    args.update(ckpt["hparams"])
    args.device = device
    ensure_dataset_defaults(args)
    ensure_setup(args, setup)
    model = build_pgm(args).to(device)
    model.load_state_dict(ckpt["ema_model_state_dict"])
    model.eval()
    return model


def load_vae_from_checkpoint(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = load_checkpoint(path)
    args = Hparams()
    args.update(ckpt["hparams"])
    args.device = device
    ensure_dataset_defaults(args)
    ensure_vae_defaults(args)
    model = HVAE(args).to(device)
    model.load_state_dict(ckpt["ema_model_state_dict"])
    model.eval()
    return model


@functools.lru_cache(maxsize=1)
def load_visualizer_bundle(checkpoint_path: str, accelerator: str) -> Tuple[Hparams, torch.nn.Module]:
    final_ckpt = load_checkpoint(checkpoint_path)
    args = Hparams()
    args.update(final_ckpt["hparams"])
    args.device = select_device(accelerator)
    ensure_dataset_defaults(args)
    ensure_vae_defaults(args)
    ensure_setup(args, "sup_cf")

    predictor_path = getattr(args, "predictor_path", "")
    pgm_path = getattr(args, "pgm_path", "")
    vae_path = getattr(args, "vae_path", "")
    missing = [name for name, value in {
        "predictor_path": predictor_path,
        "pgm_path": pgm_path,
        "vae_path": vae_path,
    }.items() if not value]
    if missing:
        raise RuntimeError(
            "The checkpoint hparams do not contain the component checkpoint paths "
            f"needed to reconstruct the visualizer: {', '.join(missing)}."
        )

    predictor = load_pgm_from_checkpoint(predictor_path, args.device, setup="sup_aux")
    pgm = load_pgm_from_checkpoint(pgm_path, args.device, setup="sup_pgm")
    vae = load_vae_from_checkpoint(vae_path, args.device)

    model = DSCM(args, pgm, predictor, vae).to(args.device)
    model.load_state_dict(final_ckpt["model_state_dict"])
    ema = EMA(model, beta=getattr(args, "ema_rate", 0.999))
    ema.ema_model.load_state_dict(final_ckpt["ema_model_state_dict"])
    ema.ema_model.to(args.device)
    ema.ema_model.eval()
    return args, ema.ema_model


def predictor_to_parents(preds: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    thickness = preds["thickness"].detach().view(-1, 1).clamp(-1, 1)
    intensity = preds["intensity"].detach().view(-1, 1).clamp(-1, 1)
    digit_idx = preds["digit"].detach().argmax(dim=-1)
    digit = F.one_hot(digit_idx, num_classes=10).float()
    return {
        "thickness": thickness.to(device),
        "intensity": intensity.to(device),
        "digit": digit.to(device),
    }


def predict_image_parents(
    model: torch.nn.Module, image: torch.Tensor, device: torch.device
) -> Dict[str, torch.Tensor]:
    intensity_loc, _ = model.predictor.encoder_i(image).chunk(2, dim=-1)
    intensity = torch.tanh(intensity_loc)
    predictions = model.predictor.predict(x=image, intensity=intensity)
    return predictor_to_parents(predictions, device)


@contextlib.contextmanager
def isolated_style_rng(style_seed: int, device: torch.device):
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(style_seed))
        yield


@torch.no_grad()
def generate_from_sliders(
    args: Hparams,
    model: torch.nn.Module,
    digit: int,
    thickness: float,
    intensity: float,
    style_seed: int = 0,
) -> Tuple[Image.Image, Dict[str, float]]:
    parents = build_parent_dict(digit, thickness, intensity, args.device)
    context = vae_preprocess(args, parents, args.device)
    with isolated_style_rng(style_seed, args.device):
        x, _ = model.vae.sample(parents=context, return_loc=True)
    summary = summarize_parents(parents)
    summary["style_seed"] = int(style_seed)
    return tensor_to_pil(x), summary


@torch.no_grad()
def linked_intensity_from_thickness(
    args: Hparams,
    model: torch.nn.Module,
    thickness: float,
    seed_image: Image.Image | None = None,
) -> float:
    normalized_thickness = torch.tensor(
        [[normalize_value(thickness, "thickness")]],
        device=args.device,
        dtype=torch.float32,
    )

    if seed_image is not None:
        source = preprocess_seed_image(seed_image, args.input_res).unsqueeze(0).to(args.device)
        factual = predict_image_parents(model, source, args.device)
        linked = model.pgm.counterfactual(
            obs={k: v.clone() for k, v in factual.items()},
            intervention={"thickness": normalized_thickness},
            num_particles=1,
        )["intensity"]
    else:
        # Zero base noise is the median of the standard Normal intensity prior.
        linked = torch.zeros_like(normalized_thickness)
        for transform in model.pgm.intensity_flow:
            if hasattr(transform, "condition"):
                transform = transform.condition(normalized_thickness)
            linked = transform(linked)

    normalized_intensity = float(linked.detach().cpu().clamp(-1, 1).item())
    return round(denormalize_value(normalized_intensity, "intensity"), 3)


@torch.no_grad()
def update_linked_preview(
    args: Hparams,
    model: torch.nn.Module,
    seed_image: Image.Image | None,
    digit: int,
    thickness: float,
    style_seed: int = 0,
) -> Tuple[float, Image.Image, Dict[str, float]]:
    intensity = linked_intensity_from_thickness(
        args, model, thickness, seed_image=seed_image
    )
    image, factors = generate_from_sliders(
        args, model, digit, thickness, intensity, style_seed
    )
    return intensity, image, factors


@torch.no_grad()
def predict_seed_factors(
    args: Hparams, model: torch.nn.Module, image: Image.Image
) -> Tuple[gr.Update, gr.Update, gr.Update, Dict[str, float]]:
    if image is None:
        raise gr.Error("Upload a seed image first.")
    x = preprocess_seed_image(image, args.input_res).unsqueeze(0).to(args.device)
    parents = predict_image_parents(model, x, args.device)
    summary = summarize_parents(parents)
    return (
        gr.update(value=str(summary["digit"])),
        gr.update(value=summary["thickness"]),
        gr.update(value=summary["intensity"]),
        summary,
    )


@torch.no_grad()
def render_counterfactual(
    args: Hparams,
    model: torch.nn.Module,
    image: Image.Image,
    digit: int,
    thickness: float,
    intensity: float,
) -> Tuple[Image.Image, Image.Image, Dict[str, float], Dict[str, float]]:
    target = build_parent_dict(digit, thickness, intensity, args.device)

    if image is None:
        source_image, factual_summary = generate_from_sliders(
            args, model, digit, thickness, intensity
        )
        source_tensor = preprocess_seed_image(source_image, args.input_res).unsqueeze(0).to(args.device)
        factual = target
    else:
        source_image = image.convert("L")
        source_tensor = preprocess_seed_image(source_image, args.input_res).unsqueeze(0).to(args.device)
        factual = predict_image_parents(model, source_tensor, args.device)
        factual_summary = summarize_parents(factual)

    cf_parents = model.pgm.counterfactual(
        obs={k: v.clone() for k, v in factual.items()},
        intervention={k: v.clone() for k, v in target.items()},
        num_particles=1,
    )
    factual_context = vae_preprocess(args, factual, args.device)
    cf_context = vae_preprocess(args, cf_parents, args.device)
    latents = model.vae.abduct(source_tensor, parents=factual_context, t=1.0)
    latents = [z["z"] if isinstance(z, dict) else z for z in latents]
    rec_x, rec_scale = model.vae.forward_latents(latents, parents=factual_context)
    cf_x, cf_scale = model.vae.forward_latents(latents, parents=cf_context)
    u = (source_tensor - rec_x) / rec_scale.clamp(min=1e-12)
    cf_x = torch.clamp(cf_x + cf_scale * u, min=-1.0, max=1.0)
    return (
        tensor_to_pil(source_tensor),
        tensor_to_pil(cf_x),
        factual_summary,
        summarize_parents(cf_parents),
    )


def build_app(args: Hparams, model: torch.nn.Module) -> gr.Blocks:
    with gr.Blocks(title="MorphoMNIST Causal Visualizer", css=APP_CSS) as demo:
        gr.Markdown(
            """
            # MorphoMNIST Causal Visualizer

            Control the MorphoMNIST causal factors directly or upload a seed digit and
            render a counterfactual edit from the trained final checkpoint.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                digit = gr.Radio(
                    choices=[str(i) for i in range(10)],
                    value="3",
                    label="Digit",
                )
                thickness = gr.Slider(
                    minimum=MORPHO_MIN_MAX["thickness"][0],
                    maximum=MORPHO_MIN_MAX["thickness"][1],
                    value=3.5,
                    step=0.01,
                    label="Thickness",
                )
                intensity = gr.Slider(
                    minimum=MORPHO_MIN_MAX["intensity"][0],
                    maximum=MORPHO_MIN_MAX["intensity"][1],
                    value=160.0,
                    step=0.5,
                    label="Intensity",
                )
                style_seed = gr.Slider(
                    minimum=0,
                    maximum=999999,
                    value=0,
                    step=1,
                    label="Style seed",
                )
                seed_image = gr.Image(
                    type="pil",
                    label="Seed image for counterfactual editing",
                    image_mode="L",
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate from sliders", variant="primary")
                    cf_btn = gr.Button("Render counterfactual", variant="secondary")
                load_seed_btn = gr.Button("Load seed factors from image")

            with gr.Column(scale=1):
                generated = gr.Image(
                    type="pil",
                    label="Generated image",
                    width="100%",
                    height=480,
                    elem_id="generated-preview",
                )
                original = gr.Image(type="pil", label="Seed / factual image")
                counterfactual = gr.Image(type="pil", label="Counterfactual image")
                factual_json = gr.JSON(label="Factual factors")
                target_json = gr.JSON(label="Target factors")
                slider_json = gr.JSON(label="Generated slider factors")

        generate_btn.click(
            fn=lambda d, t, i, s: generate_from_sliders(
                args, model, int(d), t, i, int(s)
            ),
            inputs=[digit, thickness, intensity, style_seed],
            outputs=[generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        thickness.release(
            fn=lambda img, d, t, s: update_linked_preview(
                args, model, img, int(d), t, int(s)
            ),
            inputs=[seed_image, digit, thickness, style_seed],
            outputs=[intensity, generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        intensity.release(
            fn=lambda d, t, i, s: generate_from_sliders(
                args, model, int(d), t, i, int(s)
            ),
            inputs=[digit, thickness, intensity, style_seed],
            outputs=[generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        digit.change(
            fn=lambda d, t, i, s: generate_from_sliders(
                args, model, int(d), t, i, int(s)
            ),
            inputs=[digit, thickness, intensity, style_seed],
            outputs=[generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        style_seed.release(
            fn=lambda d, t, i, s: generate_from_sliders(
                args, model, int(d), t, i, int(s)
            ),
            inputs=[digit, thickness, intensity, style_seed],
            outputs=[generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

        load_seed_btn.click(
            fn=lambda img: predict_seed_factors(args, model, img),
            inputs=[seed_image],
            outputs=[digit, thickness, intensity, factual_json],
            api_name=False,
        )

        cf_btn.click(
            fn=lambda img, d, t, i: render_counterfactual(
                args, model, img, int(d), t, i
            ),
            inputs=[seed_image, digit, thickness, intensity],
            outputs=[original, counterfactual, factual_json, target_json],
            api_name=False,
        )

        demo.load(
            fn=lambda img, d, t, s: update_linked_preview(
                args, model, img, int(d), t, int(s)
            ),
            inputs=[seed_image, digit, thickness, style_seed],
            outputs=[intensity, generated, slider_json],
            api_name=False,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            concurrency_id="slider-preview",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MorphoMNIST causal visualizer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="gs://medical-airnd/causal-gen/checkpoints/t_i_d/cf_torch-gpu-g4_17-07-2026",
        help="Path to the final counterfactual checkpoint.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference accelerator.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    seed_all(cli.seed, deterministic=True)
    args, model = load_visualizer_bundle(cli.checkpoint, cli.accelerator)
    app = build_app(args, model)
    app.launch(
        server_name=cli.server_name,
        server_port=cli.server_port,
        share=cli.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
