"""GPU backend: SDXL (RealVisXL) + ControlNet-Canny + IP-Adapter + CLIPSeg masks.
Runs on any 15-16GB CUDA GPU (Colab T4, Kaggle T4/P100) using fp16 + model CPU offload + VAE tiling (~9-12 GB peak)."""
import gc, ctypes, io, os, threading
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less VRAM fragmentation on T4
import cv2, numpy as np, torch
from PIL import Image, ImageFilter
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from diffusers import (AutoencoderKL, ControlNetModel, DPMSolverMultistepScheduler,
                       StableDiffusionXLControlNetInpaintPipeline)
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor, CLIPVisionModelWithProjection

TOKEN = os.getenv("API_TOKEN", "").strip()
BASE = os.getenv("BASE_MODEL", "SG161222/RealVisXL_V4.0")   # swap for any SDXL checkpoint
CN = os.getenv("CONTROLNET", "diffusers/controlnet-canny-sdxl-1.0-small")
MAX_SIDE = int(os.getenv("MAX_SIDE", "896"))        # lower (768) if you ever hit CUDA out-of-memory
OFFLOAD = os.getenv("OFFLOAD", "0") == "1"           # 1 = CPU offload (less VRAM, but ~12 GB of system RAM)
dtype = torch.float16
lock = threading.Lock()
app = FastAPI(title="Image Edit Backend")

print("Loading models...")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
cnet = ControlNetModel.from_pretrained(CN, torch_dtype=dtype)
# lighter ViT-H image encoder (~1.3 GB fp16) instead of the 3.7 GB bigG one; pairs with the vit-h adapter below
enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=dtype)
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    BASE, controlnet=cnet, vae=vae, image_encoder=enc, torch_dtype=dtype, variant="fp16",
    use_safetensors=True, low_cpu_mem_usage=True)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.bin")
if OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")          # weights live on the GPU, so system RAM is freed (~3 GB instead of ~12 GB)
pipe.vae.enable_tiling()
seg_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
seg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to("cuda").eval()


def free_mem():
    """Release Python, CUDA and glibc-held memory after every request."""
    gc.collect(); torch.cuda.empty_cache()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


free_mem()
print("Ready.")

QUALITY = ", photorealistic, natural lighting, consistent shadows and perspective, sharp detail, 8k photo"
NEG = "cartoon, painting, illustration, blurry, deformed, low quality, watermark, text, artifacts, seams"


def fit(img: Image.Image):
    w, h = img.size
    s = min(1.0, MAX_SIDE / max(w, h))
    return img.resize((max(64, int(w * s) // 8 * 8), max(64, int(h * s) // 8 * 8)), Image.LANCZOS)


def text_mask(img, text, thr=0.35, grow=0.03):
    inp = seg_proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        m = torch.sigmoid(seg(**inp).logits).squeeze().float().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    m = (cv2.resize(m, img.size) > thr).astype(np.uint8) * 255
    if m.sum() == 0:  # fallback: central box so "add" still works
        w, h = img.size
        m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    k = max(3, int(grow * max(img.size))) | 1
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    return Image.fromarray(m)


def canny(img, mask, keep_inside):
    e = cv2.Canny(np.array(img.convert("L")), 80, 180)
    if not keep_inside:                       # free the masked area so new content isn't constrained
        e[np.array(mask) > 127] = 0
    return Image.fromarray(np.stack([e] * 3, -1))


@app.get("/health")
def health():
    info = {"ok": True, "gpu": torch.cuda.get_device_name(0),
            "vram_used_gb": round(torch.cuda.memory_allocated() / 2**30, 1)}
    try:
        import psutil
        info["ram_used_gb"] = round(psutil.virtual_memory().used / 2**30, 1)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception: pass
    return info


@app.get("/check")
def check(x_api_token: str = Header("")):
    if TOKEN and x_api_token.strip() != TOKEN:
        raise HTTPException(401, "bad token")
    return {"ok": True, "token_required": bool(TOKEN)}


@app.post("/edit")
async def edit(image: UploadFile = File(...), reference: UploadFile | None = File(None),
               op: str = Form(...), target: str = Form(""), prompt: str = Form(""),
               seed: int = Form(-1), x_api_token: str = Header("")):
    if TOKEN and x_api_token.strip() != TOKEN:
        raise HTTPException(401, "bad token")
    orig = Image.open(io.BytesIO(await image.read())).convert("RGB")
    ref = Image.open(io.BytesIO(await reference.read())).convert("RGB") if reference else None
    img = fit(orig)
    gen = torch.Generator("cuda").manual_seed(seed if seed >= 0 else int(torch.randint(0, 2**31, (1,))))

    try:
      with lock, torch.inference_mode():
        mask = text_mask(img, target or prompt)
        neg, ip_scale = NEG, 0.0
        if op == "remove":
            p, strength, cn_scale, keep, steps = "empty clean background, seamless natural texture, nothing there" + QUALITY, 0.99, 0.5, False, 35
            neg += f", {target}"
        elif op == "recolor":
            p, strength, cn_scale, keep, steps = f"{prompt}, same shape and texture, realistic material" + QUALITY, 0.85, 0.85, True, 35
        else:  # add / replace
            p, strength, cn_scale, keep, steps = f"{prompt}" + QUALITY, 0.99, 0.4, False, 40
            if op == "replace":
                neg += f", {target}"
        if ref is not None:
            ip_scale = 0.6
        pipe.set_ip_adapter_scale(ip_scale)
        out = pipe(prompt=p, negative_prompt=neg, image=img, mask_image=mask,
                   control_image=canny(img, mask, keep), strength=strength,
                   controlnet_conditioning_scale=cn_scale, num_inference_steps=steps,
                   guidance_scale=6.5, ip_adapter_image=ref or Image.new("RGB", (224, 224), "black"),
                   padding_mask_crop=48, generator=gen).images[0]
    finally:
        free_mem()

    # keep untouched pixels identical: feathered composite back onto the original
    out = out.resize(orig.size, Image.LANCZOS)
    soft = mask.resize(orig.size).filter(ImageFilter.GaussianBlur(6))
    final = Image.composite(out, orig, soft)
    buf = io.BytesIO(); final.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")
