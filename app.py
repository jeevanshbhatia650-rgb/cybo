import os
os.environ["GRADIO_SERVER_PORT"] = "7860"

import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

CHROMA_PATH = "./chroma_store_export"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="nslkdd_attacks")

embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model.to(device)


def ask_llm(prompt, max_new_tokens=300):
    formatted = llm_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = llm_tokenizer(formatted, return_tensors="pt").to(device)
    output = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id
    )
    return llm_tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def retrieve_similar_attacks(query_text, top_k=5):
    query_vector = embed_model.encode(query_text).tolist()
    return collection.query(query_embeddings=[query_vector], n_results=top_k)


def build_rag_prompt(query_text, retrieved_results):
    blocks = []
    for i in range(len(retrieved_results["documents"][0])):
        text = retrieved_results["documents"][0][i]
        label = retrieved_results["metadatas"][0][i]["label"]
        distance = retrieved_results["distances"][0][i]
        blocks.append(
            "[Historical record #" + str(i + 1) + " | label: " + str(label) +
            " | distance: " + format(distance, ".4f") + "]\n" + text
        )
    context_str = "\n\n".join(blocks)
    prompt = (
        "You are a network security analyst. Below are similar historical network "
        "connection records, each with its known classification.\n\n"
        + context_str +
        "\n\nNow analyze this NEW connection, NOT in the historical database:\n\""
        + query_text +
        "\"\n\nIs this likely \"normal\" or \"anomaly\"? State your classification "
        "clearly, then explain your reasoning."
    )
    return prompt


def build_baseline_prompt(query_text):
    prompt = (
        "You are a network security analyst. Analyze this network connection record:\n\""
        + query_text +
        "\"\n\nIs this likely \"normal\" or \"anomaly\"? State your classification "
        "clearly, then explain your reasoning."
    )
    return prompt


def extract_verdict(text):
    match = re.search(r"\b(normal|anomaly)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "UNCLEAR"


def esc(s):
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def term(lines, alive=True, tone_first_word=True):
    """Render a list of (text, css_class) tuples or plain strings into a terminal body."""
    rendered = []
    for line in lines:
        if isinstance(line, tuple):
            text, cls = line
            rendered.append('<span class="' + cls + '">' + esc(text) + '</span>')
        else:
            rendered.append(esc(line))
    cursor = '<span class="cursor"></span>' if alive else ''
    body = "<br>".join(rendered) + cursor
    return '<div class="term-body">' + body + '</div>'


def verdict_html(state, label=None):
    """state: 'idle' | 'processing' | 'normal' | 'anomaly' | 'error' | 'unclear'"""
    if state == "idle":
        return '<div class="verdict-bar idle">AWAITING INPUT</div>'
    if state == "processing":
        return '<div class="verdict-bar processing">⏳ ANALYZING CONNECTION…</div>'
    if state == "error":
        return '<div class="verdict-bar error">✕ PIPELINE ERROR — SEE LOGS</div>'
    if state == "normal":
        return '<div class="verdict-bar normal">✓ VERDICT: NORMAL</div>'
    if state == "anomaly":
        return '<div class="verdict-bar anomaly">⚠ VERDICT: ANOMALY DETECTED</div>'
    return '<div class="verdict-bar unclear">? VERDICT: UNCLEAR</div>'


def classify_connection(query_text):
    if not query_text.strip():
        yield (
            term([("[ERROR] no input provided", "err")], False),
            term([], False),
            term([], False),
            verdict_html("idle"),
        )
        return

    retrieval_log = [
        "root@rag-engine:~$ query received",
        "> \"" + query_text[:60] + "...\"",
        ("[INIT] embedding query...", "dim"),
    ]
    baseline_log = [("root@baseline-llm:~$ standing by...", "dim")]
    rag_log = [("root@rag-llm:~$ standing by...", "dim")]
    yield term(retrieval_log), term(baseline_log), term(rag_log), verdict_html("processing")

    try:
        retrieved = retrieve_similar_attacks(query_text, top_k=5)
        retrieval_log.append(("[OK] embedding complete", "ok"))
        retrieval_log.append(("[QUERY] searching vector store (chroma)...", "dim"))
        for i in range(len(retrieved["documents"][0])):
            label = retrieved["metadatas"][0][i]["label"]
            dist = retrieved["distances"][0][i]
            snippet = retrieved["documents"][0][i][:70].replace("\n", " ")
            retrieval_log.append(
                "  #" + str(i + 1) + " [" + str(label) + "] dist=" +
                format(dist, ".2f") + " :: " + snippet + "..."
            )
        retrieval_log.append(("[DONE] retrieval complete", "ok"))
    except Exception as e:
        retrieval_log.append(("[FATAL] " + str(e), "err"))
        yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), verdict_html("error")
        return

    yield term(retrieval_log), term(baseline_log), term(rag_log), verdict_html("processing")

    baseline_log = [("root@baseline-llm:~$ generating (no context)...", "dim")]
    yield term(retrieval_log), term(baseline_log), term(rag_log), verdict_html("processing")
    try:
        baseline = ask_llm(build_baseline_prompt(query_text))
        baseline_log.append(("[DONE]", "warn"))
        baseline_log.append(baseline)
    except Exception as e:
        baseline_log.append(("[FATAL] " + str(e), "err"))
        yield term(retrieval_log), term(baseline_log, False), term(rag_log, False), verdict_html("error")
        return

    yield term(retrieval_log), term(baseline_log), term(rag_log), verdict_html("processing")

    rag_log = [("root@rag-llm:~$ generating (with retrieved context)...", "dim")]
    yield term(retrieval_log), term(baseline_log), term(rag_log), verdict_html("processing")
    try:
        rag_response = ask_llm(build_rag_prompt(query_text, retrieved))
        rag_log.append(("[DONE]", "ok"))
        rag_log.append(rag_response)
    except Exception as e:
        rag_log.append(("[FATAL] " + str(e), "err"))
        yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), verdict_html("error")
        return

    verdict = extract_verdict(rag_response)
    state = {"NORMAL": "normal", "ANOMALY": "anomaly"}.get(verdict, "unclear")

    yield (
        term(retrieval_log, False),
        term(baseline_log, False),
        term(rag_log, False),
        verdict_html(state),
    )


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --void: #06070a;
  --panel-border: #1c2128;
  --cyan: #00f0c4;
  --amber: #ffb020;
  --violet: #7c5cff;
  --phosphor: #39ff88;
  --red: #ff4d4d;
  --text-dim: #6b7280;
  --text-mid: #9ca3af;
}

body, .gradio-container {
  background: var(--void) !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* fixed animated backdrop canvas, injected once via JS below */
#bg-canvas {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
}
.bg-grad {
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background:
    radial-gradient(ellipse 60% 50% at 50% 0%, rgba(0,240,196,0.08), transparent 60%),
    radial-gradient(ellipse 80% 60% at 50% 100%, rgba(124,92,255,0.10), transparent 70%);
}
.scanlines {
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background: repeating-linear-gradient(to bottom, rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px, transparent 1px, transparent 3px);
  mix-blend-mode: overlay;
}

.gradio-container, .gradio-container * { position: relative; z-index: 5; }

/* header */
.console-header {
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 12px; margin-bottom: 6px;
}
.brand-row { display: flex; align-items: center; gap: 14px; }
.brand-mark {
  width: 36px; height: 36px; border: 1.5px solid var(--cyan); border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 18px rgba(0,240,196,0.35), inset 0 0 12px rgba(0,240,196,0.15);
  flex-shrink: 0;
}
.brand-mark::before {
  content: ''; width: 13px; height: 13px; border-radius: 50%;
  background: var(--cyan); box-shadow: 0 0 10px var(--cyan);
  animation: pulse-dot 2.4s ease-in-out infinite;
}
@keyframes pulse-dot { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:.5; transform:scale(.8); } }
.brand-title {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 20px; font-weight: 700; letter-spacing: .04em; color: #f3f4f6; line-height: 1.1;
}
.brand-sub { font-size: 11px; color: var(--text-dim); letter-spacing: .12em; text-transform: uppercase; margin-top: 3px; }
.status-pill {
  display: inline-flex; align-items: center; gap: 8px; font-size: 11px;
  letter-spacing: .08em; text-transform: uppercase; color: var(--cyan);
  border: 1px solid rgba(0,240,196,.3); background: rgba(0,240,196,.06);
  padding: 7px 14px; border-radius: 100px;
}
.status-pill .dot { width:6px; height:6px; border-radius:50%; background: var(--cyan); box-shadow:0 0 8px var(--cyan); animation: pulse-dot 1.8s ease-in-out infinite; display:inline-block; }

/* input area */
.console-input-label {
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--text-mid); margin-bottom: 6px;
}
.console-input-label::before { content: '>_ '; color: var(--cyan); }

textarea, input[type="text"] {
  background: #050709 !important;
  color: var(--phosphor) !important;
  font-family: 'JetBrains Mono', monospace !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 10px !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 3px rgba(0,240,196,.12) !important;
}

button {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: .05em !important;
  text-transform: uppercase !important;
  border-radius: 9px !important;
}
button.primary, .run-btn-row button {
  color: #06070a !important;
  background: linear-gradient(135deg, var(--cyan), #00c9a7) !important;
  border: none !important;
  box-shadow: 0 0 24px rgba(0,240,196,.35) !important;
}

/* verdict bar */
.verdict-bar {
  display: flex; align-items: center; justify-content: center; gap: 12px;
  padding: 16px; border-radius: 12px; margin: 10px 0 4px;
  border: 1px solid var(--panel-border); background: rgba(12,15,20,.6);
  font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 15px;
  letter-spacing: .08em; text-transform: uppercase; transition: all .3s;
}
.verdict-bar.idle { color: var(--text-dim); }
.verdict-bar.processing { color: var(--amber); border-color: rgba(255,176,32,.4); }
.verdict-bar.normal { color: var(--phosphor); border-color: rgba(57,255,136,.4); box-shadow: 0 0 30px -10px rgba(57,255,136,.4); }
.verdict-bar.anomaly { color: var(--red); border-color: rgba(255,77,77,.45); box-shadow: 0 0 30px -10px rgba(255,77,77,.45); }
.verdict-bar.error { color: var(--red); border-color: rgba(255,77,77,.45); }
.verdict-bar.unclear { color: var(--amber); border-color: rgba(255,176,32,.4); }

/* terminal windows */
.term-titlebar {
  display: flex; align-items: center; gap: 10px; padding: 10px 14px;
  background: linear-gradient(180deg, #11151b, #0c0f14);
  border: 1px solid var(--panel-border); border-bottom: none;
  border-radius: 12px 12px 0 0;
  font-size: 11px; color: var(--text-mid); letter-spacing: .03em;
}
.term-dots { display: flex; gap: 6px; }
.term-dots span { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.term-dots span:nth-child(1){ background:#ff5f56; }
.term-dots span:nth-child(2){ background:#ffbd2e; }
.term-dots span:nth-child(3){ background:#27c93f; }
.term-name .accent.violet { color: var(--violet); }
.term-name .accent.amber { color: var(--amber); }
.term-name .accent.cyan { color: var(--cyan); }

.term-body {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12.5px; line-height: 1.6;
  padding: 14px; height: 260px; overflow-y: auto;
  white-space: pre-wrap; word-break: break-word;
  background: rgba(8,10,13,.85) !important;
  border: 1px solid var(--panel-border);
  border-top: none;
  border-radius: 0 0 12px 12px;
}
.term-body::-webkit-scrollbar { width: 6px; }
.term-body::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 3px; }
.term-body .dim { color: var(--text-dim); }
.term-body .ok { color: var(--cyan); }
.term-body .err { color: var(--red); }
.term-body .warn { color: var(--amber); }

.cursor {
  display: inline-block; width: 7px; height: 14px;
  background: var(--phosphor); vertical-align: middle; margin-left: 2px;
  animation: blink 1s step-start infinite;
}
@keyframes blink { 50% { opacity: 0; } }

.footer-note {
  text-align: center; margin-top: 18px; font-size: 11px;
  color: var(--text-dim); letter-spacing: .05em;
}
"""

HEADER_HTML = """
<canvas id="bg-canvas"></canvas>
<div class="bg-grad"></div>
<div class="scanlines"></div>

<div class="console-header">
  <div class="brand-row">
    <div class="brand-mark"></div>
    <div>
      <div class="brand-title">INTRUSION CONSOLE</div>
      <div class="brand-sub">RAG-Augmented Network Anomaly Analyst</div>
    </div>
  </div>
  <div class="status-pill"><span class="dot"></span> ENGINE ONLINE</div>
</div>
"""

TERM_HEADER_RETRIEVAL = """
<div class="term-titlebar">
  <div class="term-dots"><span></span><span></span><span></span></div>
  <div class="term-name">rag-engine:~/<span class="accent violet">retrieval.log</span></div>
</div>
"""

TERM_HEADER_BASELINE = """
<div class="term-titlebar">
  <div class="term-dots"><span></span><span></span><span></span></div>
  <div class="term-name">baseline-llm:~/<span class="accent amber">output.log</span></div>
</div>
"""

TERM_HEADER_RAG = """
<div class="term-titlebar">
  <div class="term-dots"><span></span><span></span><span></span></div>
  <div class="term-name">rag-llm:~/<span class="accent cyan">output.log</span></div>
</div>
"""

BG_SCRIPT = """
<script>
(function() {
  function init() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas || canvas.dataset.initd) { return; }
    canvas.dataset.initd = "1";
    const ctx = canvas.getContext('2d');
    let W, H;
    function resize() {
      W = canvas.width = window.innerWidth;
      H = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    const RADIUS = Math.min(window.innerWidth, window.innerHeight) * 0.4;
    let angleY = 0;
    const angleX = 0.45;

    function project(x, y, z, cx, cy) {
      const fov = 700;
      const factor = fov / (fov + z);
      return [cx + x * factor, cy + y * factor];
    }
    function rotate(x, y, z, ay, ax) {
      let cosY = Math.cos(ay), sinY = Math.sin(ay);
      let x1 = x * cosY - z * sinY;
      let z1 = x * sinY + z * cosY;
      let cosX = Math.cos(ax), sinX = Math.sin(ax);
      let y1 = y * cosX - z1 * sinX;
      let z2 = y * sinX + z1 * cosX;
      return [x1, y1, z2];
    }

    const LAT_STEPS = 12, LON_STEPS = 20;

    function drawGlobe() {
      const cx = W * 0.8, cy = H * 0.3;
      ctx.lineWidth = 1;
      for (let i = 0; i <= LAT_STEPS; i++) {
        const lat = (i / LAT_STEPS) * Math.PI - Math.PI / 2;
        ctx.beginPath();
        let started = false;
        for (let j = 0; j <= LON_STEPS; j++) {
          const lon = (j / LON_STEPS) * Math.PI * 2;
          const x = RADIUS * Math.cos(lat) * Math.cos(lon);
          const y = RADIUS * Math.sin(lat);
          const z = RADIUS * Math.cos(lat) * Math.sin(lon);
          const [rx, ry, rz] = rotate(x, y, z, angleY, angleX);
          const [px, py] = project(rx, ry, rz, cx, cy);
          if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
        }
        ctx.strokeStyle = 'rgba(0,240,196,0.12)';
        ctx.stroke();
      }
      for (let j = 0; j < LON_STEPS; j++) {
        const lon = (j / LON_STEPS) * Math.PI * 2;
        ctx.beginPath();
        let started = false;
        for (let i = 0; i <= LAT_STEPS; i++) {
          const lat = (i / LAT_STEPS) * Math.PI - Math.PI / 2;
          const x = RADIUS * Math.cos(lat) * Math.cos(lon);
          const y = RADIUS * Math.sin(lat);
          const z = RADIUS * Math.cos(lat) * Math.sin(lon);
          const [rx, ry, rz] = rotate(x, y, z, angleY, angleX);
          const [px, py] = project(rx, ry, rz, cx, cy);
          if (!started) { ctx.moveTo(px, py); started = true; } else { ctx.lineTo(px, py); }
        }
        ctx.strokeStyle = 'rgba(124,92,255,0.08)';
        ctx.stroke();
      }
    }

    const cols = [];
    const COL_COUNT = 22;
    for (let i = 0; i < COL_COUNT; i++) {
      cols.push({ x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight, speed: 0.4 + Math.random() * 1.1, len: 4 + Math.floor(Math.random() * 7) });
    }

    function drawColumns() {
      ctx.font = '11px JetBrains Mono, monospace';
      for (const c of cols) {
        for (let k = 0; k < c.len; k++) {
          const yy = c.y - k * 14;
          if (yy < -20 || yy > H + 20) continue;
          const alpha = Math.max(0, 0.20 - k * 0.02);
          ctx.fillStyle = 'rgba(57,255,136,' + alpha + ')';
          const ch = (Math.floor((c.x + k + Date.now() * 0.001)) % 2) === 0 ? '0' : '1';
          ctx.fillText(ch, c.x, yy);
        }
        c.y += c.speed;
        if (c.y > H + 100) { c.y = -50; c.x = Math.random() * W; }
      }
    }

    function tick() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = '#06070a';
      ctx.fillRect(0, 0, W, H);
      drawColumns();
      drawGlobe();
      angleY += 0.0022;
      requestAnimationFrame(tick);
    }
    tick();
  }
  // Gradio mounts async — retry until canvas exists
  const iv = setInterval(function() {
    if (document.getElementById('bg-canvas')) { init(); clearInterval(iv); }
  }, 200);
})();
</script>
"""

with gr.Blocks(title="Intrusion Console", css=CSS) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Group(elem_classes="input-panel"):
        gr.HTML('<div class="console-input-label">DESCRIBE CONNECTION</div>')
        inp = gr.Textbox(
            show_label=False,
            lines=3,
            placeholder='e.g. A tcp connection using the http service, flag S0, 0 bytes sent, 95% SYN error rate, 480 connections to same host in 2s...'
        )
        with gr.Row(elem_classes="run-btn-row"):
            btn = gr.Button("Execute Analysis", variant="primary")

    verdict_box = gr.HTML(verdict_html("idle"))

    with gr.Row():
        with gr.Column():
            gr.HTML(TERM_HEADER_RETRIEVAL)
            retrieval_box = gr.HTML(term([("root@rag-engine:~$ standing by...", "dim")], False))
        with gr.Column():
            gr.HTML(TERM_HEADER_BASELINE)
            baseline_box = gr.HTML(term([("root@baseline-llm:~$ standing by...", "dim")], False))
        with gr.Column():
            gr.HTML(TERM_HEADER_RAG)
            rag_box = gr.HTML(term([("root@rag-llm:~$ standing by...", "dim")], False))

    gr.HTML('<div class="footer-note">NSL-KDD VECTOR STORE · QWEN2.5-7B-INSTRUCT · CHROMADB</div>')
    gr.HTML(BG_SCRIPT)

    btn.click(
        classify_connection,
        inputs=inp,
        outputs=[retrieval_box, baseline_box, rag_box, verdict_box]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
