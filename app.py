import io, json, os, sqlite3, time, uuid
import requests, streamlit as st
from contextlib import nullcontext
from PIL import Image, ImageOps

st.set_page_config(page_title="PixelChat", page_icon="🎨", layout="wide")

def cfg(k, d=""):
    try: return st.secrets.get(k, os.getenv(k, d))
    except Exception: return os.getenv(k, d)

API_TOKEN = str(cfg("API_TOKEN")).strip()
GROQ_KEY, LLM = cfg("GROQ_API_KEY"), cfg("LLM_MODEL", "openai/gpt-oss-120b")
os.makedirs("data/img", exist_ok=True)

# ---------- storage (SQLite) ----------
db = sqlite3.connect("data/chats.db", check_same_thread=False)
db.executescript("""
CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, title TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS msgs(id INTEGER PRIMARY KEY AUTOINCREMENT, chat TEXT, role TEXT, content TEXT, img TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS versions(id INTEGER PRIMARY KEY AUTOINCREMENT, chat TEXT, path TEXT, label TEXT, ts REAL);
""")
q = lambda s, *a: db.execute(s, a).fetchall()
def run(s, *a): db.execute(s, a); db.commit()
def kv_get(k, d=""): r = q("SELECT v FROM kv WHERE k=?", k); return r[0][0] if r else d
def kv_set(k, v): run("INSERT OR REPLACE INTO kv VALUES(?,?)", k, v)
def load_img(f, cap=2048):                       # fix phone-photo rotation + shrink before upload
    im = ImageOps.exif_transpose(Image.open(f)).convert("RGB"); im.thumbnail((cap, cap), Image.LANCZOS); return im
def save_img(img): p = f"data/img/{uuid.uuid4().hex}.png"; img.save(p); return p
def new_chat():
    cid = uuid.uuid4().hex[:8]; run("INSERT INTO chats VALUES(?,?,?)", cid, "New chat", time.time()); return cid
def current(cid):
    r = q("SELECT path,label FROM versions WHERE chat=? ORDER BY id DESC LIMIT 1", cid); return r[0] if r else None

# ---------- agent ----------
SYSTEM = """You are PixelChat, a friendly image-editing assistant. Understand what the user wants and reply ONLY with JSON:
{"reply": "<short friendly message>", "edit": null | {"op": "add|remove|replace|recolor", "target": "<1-3 word lowercase noun to detect, e.g. face, car, sofa, shirt>", "prompt": "<photoreal description of ONLY the new content>", "region": null | [x0,y0,x1,y1]}}
- target must be a short detectable noun phrase (never a sentence). For remove/replace/recolor it is the object itself. For add it is the anchor object the new item attaches to.
- region is ONLY for op=add: where the new item goes, as fractions of the target's bounding box (0,0 = top-left, 1,1 = bottom-right; values may go outside 0-1).
  Examples: sunglasses -> target "face", region [0.02,0.27,0.98,0.55]; hat -> target "face", region [-0.15,-0.7,1.15,0.15]; a cat on a sofa -> target "sofa", region [0.2,-0.4,0.8,0.5].
- prompt describes only what appears in the edited area (e.g. "dark aviator sunglasses worn on the face"), never the whole image. recolor example: 'a deep red leather jacket'. remove: prompt may be empty.
- Use edit=null for chit-chat, questions, or when you need clarification (ask ONE question). If a reference image is attached, describe it generically (e.g. 'a vintage leather handbag like the reference').
- One edit per message. If no image is loaded yet, tell the user to upload one."""

def agent(cid, user_text, has_img, has_ref):
    hist = [{"role": r, "content": c} for r, c in q("SELECT role,content FROM msgs WHERE chat=? ORDER BY id DESC LIMIT 11", cid)][::-1][:-1]
    ctx = f"[state: image_loaded={has_img}, reference_attached={has_ref}]"
    msgs = [{"role": "system", "content": SYSTEM}] + hist + [{"role": "user", "content": f"{ctx}\n{user_text}"}]
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=60,
        headers={"Authorization": f"Bearer {GROQ_KEY}"},
        json={"model": LLM, "messages": msgs, "temperature": 0.3, "response_format": {"type": "json_object"},
              **({"reasoning_effort": "low"} if "gpt-oss" in LLM else {})})
    if not r.ok: raise RuntimeError(f"Groq {r.status_code}: {r.text[:300]}")
    return json.loads(r.json()["choices"][0]["message"]["content"])

def call_backend(base_path, ref_path, e):
    with open(base_path, "rb") as f1, (open(ref_path, "rb") if ref_path else nullcontext()) as f2:
        files = {"image": f1}
        if f2: files["reference"] = f2
        r = requests.post(f"{BACKEND}/edit", files=files, timeout=900,
            headers={"x-api-token": API_TOKEN, "ngrok-skip-browser-warning": "1"},
            data={"op": e["op"], "target": e.get("target", ""), "prompt": e.get("prompt", ""),
                  "region": json.dumps(e["region"]) if e.get("region") else ""})
    if not r.ok: raise RuntimeError(f"Backend {r.status_code}: {r.text[:300]}")
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# ---------- sidebar: chats ----------
if "cid" not in st.session_state:
    r = q("SELECT id FROM chats ORDER BY ts DESC LIMIT 1"); st.session_state.cid = r[0][0] if r else new_chat()
with st.sidebar:
    st.title("🎨 PixelChat")
    if st.button("＋ New chat", width="stretch"): st.session_state.cid = new_chat(); st.rerun()
    st.caption("Chats")
    for cid, title in q("SELECT id,title FROM chats ORDER BY ts DESC"):
        c1, c2 = st.columns([5, 1])
        if c1.button(("● " if cid == st.session_state.cid else "") + title[:28], key=f"s{cid}", width="stretch"):
            st.session_state.cid = cid; st.rerun()
        if c2.button("🗑", key=f"d{cid}"):
            for t in ("chats", "msgs", "versions"): run(f"DELETE FROM {t} WHERE {'id' if t=='chats' else 'chat'}=?", cid)
            st.session_state.pop("cid"); st.rerun()
    with st.expander("GPU backend", expanded=not kv_get("backend", cfg("BACKEND_URL"))):
        BACKEND = st.text_input("Backend URL (changes every Colab run)", kv_get("backend", cfg("BACKEND_URL"))).strip().rstrip("/")
        kv_set("backend", BACKEND)
        H = {"ngrok-skip-browser-warning": "1", "x-api-token": API_TOKEN}
        if not API_TOKEN: st.warning("API_TOKEN is empty - add it to .streamlit/secrets.toml and restart Streamlit")
        try:
            if not BACKEND: st.warning("Paste your ngrok URL")
            else:
                st.success(requests.get(f"{BACKEND}/health", timeout=8, headers=H).json())
                c = requests.get(f"{BACKEND}/check", timeout=8, headers=H)
                if c.ok: st.success("Token OK ✅")
                elif c.status_code == 401: st.error("Token mismatch (401): API_TOKEN here differs from the one Colab started with. Fix it, then restart the Colab session.")
                else: st.warning(f"/check returned {c.status_code} - backend may be running an old server.py")
        except Exception as ex: st.error(f"Offline: {ex}")
cid = st.session_state.cid

# ---------- main ----------
left, right = st.columns([3, 2], gap="large")
with right:
    st.subheader("Current image")
    cur = current(cid)
    if cur:
        st.image(cur[0], caption=cur[1], width="stretch")
        n = q("SELECT COUNT(*) FROM versions WHERE chat=?", cid)[0][0]
        b1, b2 = st.columns(2)
        if b1.button("↩ Undo", disabled=n < 2, width="stretch"):
            run("DELETE FROM versions WHERE id=(SELECT MAX(id) FROM versions WHERE chat=?)", cid); st.rerun()
        b2.download_button("⬇ Download", open(cur[0], "rb"), "edited.png", width="stretch")
        with st.expander(f"History ({n} versions)"):
            for p, l in q("SELECT path,label FROM versions WHERE chat=? ORDER BY id", cid): st.image(p, caption=l, width=160)
    else:
        st.info("Attach an image with the 📎 in the chat box to begin.")

with left:
    for role, content, img in q("SELECT role,content,img FROM msgs WHERE chat=? ORDER BY id", cid):
        with st.chat_message(role):
            st.write(content)
            if img: st.image(img, width=260)

    sub = st.chat_input("Describe an edit… (📎 attach base image or reference)", accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp"])
    if sub:
        text, files = (sub.text or "").strip(), list(sub.files or [])
        base_path = ref_path = None
        if files and not current(cid):                       # first upload becomes the base image
            base_path = save_img(load_img(files.pop(0)))
            run("INSERT INTO versions(chat,path,label,ts) VALUES(?,?,?,?)", cid, base_path, "Original", time.time())
        if files: ref_path = save_img(load_img(files[0]))   # later uploads = reference
        shown = ref_path or base_path
        run("INSERT INTO msgs(chat,role,content,img,ts) VALUES(?,?,?,?,?)", cid, "user", text or "(image)", shown, time.time())
        if q("SELECT COUNT(*) FROM msgs WHERE chat=?", cid)[0][0] == 1:
            run("UPDATE chats SET title=? WHERE id=?", (text or "Image chat")[:40], cid)
        try:
            res = agent(cid, text or "(uploaded an image)", bool(current(cid)), bool(ref_path))
            reply, e = res.get("reply", ""), res.get("edit")
            if e and current(cid):
                with st.spinner(f"{ {'add':'Adding','remove':'Removing','replace':'Replacing','recolor':'Recoloring'}.get(e['op'],'Editing') }… (diffusion on GPU, ~30-90s)"):
                    out = call_backend(current(cid)[0], ref_path, e)
                p = save_img(out)
                run("INSERT INTO versions(chat,path,label,ts) VALUES(?,?,?,?)", cid, p, f"{e['op']}: {e.get('target','')}"[:60], time.time())
        except Exception as ex:
            reply = f"⚠️ Something went wrong: {ex}"
        run("INSERT INTO msgs(chat,role,content,ts) VALUES(?,?,?,?)", cid, "assistant", reply, time.time())
        st.rerun()
