import os
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

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


def term(lines, alive=True):
    if alive:
        cursor = '<span class="cursor">█</span>'
    else:
        cursor = ''
    body = "<br>".join(lines) + cursor
    return '<div class="term-body">' + body + '</div>'


def classify_connection(query_text):
    if not query_text.strip():
        yield term(["[ERROR] no input provided"], False), term([], False), term([], False), "—"
        return

    retrieval_log = [
        "root@rag-engine:~$ query received",
        "> \"" + query_text[:60] + "...\"",
        "[INIT] embedding query..."
    ]
    baseline_log = ["root@baseline-llm:~$ standing by..."]
    rag_log = ["root@rag-llm:~$ standing by..."]
    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: PROCESSING"

    try:
        retrieved = retrieve_similar_attacks(query_text, top_k=5)
        retrieval_log.append("[OK] embedding complete")
        retrieval_log.append("[QUERY] searching vector store (chroma)...")
        for i in range(len(retrieved["documents"][0])):
            label = retrieved["metadatas"][0][i]["label"]
            dist = retrieved["distances"][0][i]
            snippet = retrieved["documents"][0][i][:70].replace("\n", " ")
            retrieval_log.append(
                "  #" + str(i + 1) + " [" + str(label) + "] dist=" +
                format(dist, ".2f") + " :: " + snippet + "..."
            )
        retrieval_log.append("[DONE] retrieval complete")
    except Exception as e:
        retrieval_log.append("[FATAL] " + str(e))
        yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), "## VERDICT: ERROR"
        return

    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: PROCESSING"

    baseline_log = ["root@baseline-llm:~$ generating (no context)..."]
    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: PROCESSING"
    try:
        baseline = ask_llm(build_baseline_prompt(query_text))
        baseline_log.append("[DONE]")
        baseline_log.append(baseline)
    except Exception as e:
        baseline_log.append("[FATAL] " + str(e))
        yield term(retrieval_log), term(baseline_log, False), term(rag_log, False), "## VERDICT: ERROR"
        return

    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: PROCESSING"

    rag_log = ["root@rag-llm:~$ generating (with retrieved context)..."]
    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: PROCESSING"
    try:
        rag_response = ask_llm(build_rag_prompt(query_text, retrieved))
        rag_log.append("[DONE]")
        rag_log.append(rag_response)
    except Exception as e:
        rag_log.append("[FATAL] " + str(e))
        yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), "## VERDICT: ERROR"
        return

    verdict = extract_verdict(rag_response)
    if verdict == "NORMAL":
        color = "#00ff66"
    elif verdict == "ANOMALY":
        color = "#ff3b3b"
    else:
        color = "#ffaa00"

    verdict_md = '<h2 style="color:' + color + '">VERDICT: ' + verdict + '</h2>'
    yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), verdict_md


CSS = """
body, .gradio-container { background: #050805 !important; }
.term-titlebar {
    background: #0f1a0f;
    padding: 4px 10px;
    font-family: monospace;
    color: #00ff66;
    font-size: 12px;
    border-bottom: 1px solid #00ff66;
}
.term-body {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #00ff66;
    padding: 10px;
    height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    background: #0a0f0a;
    border: 1px solid #00ff66;
    border-radius: 6px;
}
.cursor { animation: blink 1s step-start infinite; }
@keyframes blink { 50% { opacity: 0; } }
textarea, input {
    background: #0a0f0a !important;
    color: #00ff66 !important;
    font-family: monospace !important;
    border: 1px solid #00ff66 !important;
}
button {
    background: #0a0f0a !important;
    color: #00ff66 !important;
    border: 1px solid #00ff66 !important;
    font-family: monospace !important;
}
"""

with gr.Blocks(title="RAG Intrusion Console") as demo:
    gr.HTML('<h1 style="color:#00ff66;font-family:monospace;">NETWORK INTRUSION ANALYSIS CONSOLE</h1>')
    inp = gr.Textbox(label="> describe connection", lines=3)
    btn = gr.Button("EXECUTE ANALYSIS")
    verdict_box = gr.Markdown()

    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="term-titlebar">rag-engine:~/retrieval.log</div>')
            retrieval_box = gr.HTML()
        with gr.Column():
            gr.HTML('<div class="term-titlebar">baseline-llm:~/output.log</div>')
            baseline_box = gr.HTML()
        with gr.Column():
            gr.HTML('<div class="term-titlebar">rag-llm:~/output.log</div>')
            rag_box = gr.HTML()

    btn.click(
        classify_connection,
        inputs=inp,
        outputs=[retrieval_box, baseline_box, rag_box, verdict_box]
    )

demo.launch()