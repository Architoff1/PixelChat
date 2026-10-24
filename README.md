# PixelChat – chat-driven realistic image editing

Stack: Streamlit UI → Groq LLM agent (free) → FastAPI/uvicorn GPU backend (Kaggle + ngrok) → SDXL (RealVisXL) + ControlNet-Canny + IP-Adapter + CLIPSeg.
VRAM: ~9–12 GB with fp16 + CPU offload + VAE tiling (fits Kaggle T4 16 GB).

## 1. GPU backend (Google Colab, free T4)
1. Open `backend/colab_backend.ipynb` in Colab (upload it at colab.research.google.com → Upload).
2. Runtime → Change runtime type → **T4 GPU**.
3. 🔑 Secrets (left sidebar): add `API_TOKEN` (any password you invent) and `NGROK_TOKEN` (free at ngrok.com). Turn **Notebook access** ON for both.
4. Run all cells in order. Wait for `Ready.`, then copy the printed `PUBLIC URL`.
   Keep the tab open; Colab disconnects on idle and after ~12 h (the URL changes each run unless you use ngrok's free static domain).

## 2. UI
Local: `pip install -r requirements.txt` then create `.streamlit/secrets.toml`:
```
BACKEND_URL = "https://xxxx.ngrok-free.app"
API_TOKEN = "same-password"
GROQ_API_KEY = "gsk_..."   # free at console.groq.com
```
`streamlit run app.py`

Deploy free: push to GitHub → share.streamlit.io → paste the same secrets. (Its disk is ephemeral, so chat history resets on redeploy; for permanent history swap SQLite for a free Supabase/Turso DB.)

## Notes
- Ops: add, remove, replace, recolor; attach a reference image in chat to guide add/replace (IP-Adapter).
- Edits are masked + feather-composited, so untouched pixels stay identical.
- Upgrades: swap `BASE_MODEL` for any SDXL checkpoint; swap CLIPSeg for GroundingDINO+SAM for sharper masks.
