import spaces
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, re

CHROMA_PATH = "./chroma_store_export"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="nslkdd_attacks")

embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

LLM_NAME = "Qwen/Qwen2.5-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model.to(device)

@spaces.GPU
def ask_llm(prompt, max_new_tokens=300):
    formatted = llm_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = llm_tokenizer(formatted, return_tensors="pt").to(device)
    output = llm_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 temperature=0.3, do_sample=True,
                                 pad_token_id=llm_tokenizer.eos_token_id)
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
        blocks.append(f"[Historical record #{i+1} | label: {label} | distance: {distance:.4f}]\n{text}")
    context_str = "\n\n".join(blocks)
    return f"""You are a network security analyst. Below are similar historical network connection records, each with its known classification.

{context_str}

Now analyze this NEW connection, NOT in the historical database:
"{query_text}"

Is this likely "normal" or "anomaly"? State your classification clearly, then explain your reasoning."""


def build_baseline_prompt(query_text):
    return f"""You are a network security analyst. Analyze this network connection record:
"{query_text}"

Is this likely "normal" or "anomaly"? State your classification clearly, then explain your reasoning."""


def extract_verdict(text):
    match = re.search(r"\b(normal|anomaly)\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else "UNCLEAR"


def term(lines, alive=True):
    cursor = '<span class="cursor">█</span>' if alive else ''
    body = "<br>".join(lines) + cursor
    return f'<div class="term-body">{body}</div>'


def classify_connection(query_text):
    if not query_text.strip():
        yield term(["[ERROR] no input provided"], False), term([], False), term([], False), "—"
        return

    retrieval_log = [f"root@rag-engine:~$ query received", f"> \"{query_text[:60]}...\"", "[INIT] embedding query..."]
    baseline_log = ["root@baseline-llm:~$ standing by..."]
    rag_log = ["root@rag-llm:~$ standing by..."]
    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: ⏳ PROCESSING"

    try:
        retrieved = retrieve_similar_attacks(query_text, top_k=5)
        retrieval_log.append("[OK] embedding complete")
        retrieval_log.append("[QUERY] searching vector store (chroma)...")
        for i in range(len(retrieved["documents"][0])):
            label = retrieved["metadatas"][0][i]["label"]
            dist = retrieved["distances"][0][i]
            snippet = retrieved["documents"][0][i][:70].replace("\n", " ")
            retrieval_log.append(f"  #{i+1} [{label}] dist={dist:.2f} :: {snippet}...")
        retrieval_log.append("[DONE] retrieval complete")
    except Exception as e:
        retrieval_log.append(f"[FATAL] {e}")
        yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), "## VERDICT: ⚠️ ERROR"
        return

    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: ⏳ PROCESSING"

    baseline_log = ["root@baseline-llm:~$ generating (no context)..."]
    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: ⏳ PROCESSING"
    try:
        baseline = ask_llm(build_baseline_prompt(query_text))
        baseline_log.append("[DONE]")
        baseline_log.append(baseline)
    except Exception as e:
        baseline_log.append(f"[FATAL] {e}")
        yield term(retrieval_log), term(baseline_log, False), term(rag_log, False), "## VERDICT: ⚠️ ERROR"
        return

    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: ⏳ PROCESSING"

    rag_log = ["root@rag-llm:~$ generating (with retrieved context)..."]
    yield term(retrieval_log), term(baseline_log), term(rag_log), "## VERDICT: ⏳ PROCESSING"
    try:
        rag_response = ask_llm(build_rag_prompt(query_text, retrieved))
        rag_log.append("[DONE]")
        rag_log.append(rag_response)
    except Exception as e:
        rag_log.append(f"[FATAL] {e}")
        yield term(retrieval_log, False), term(baseline_log, False), term(rag_log, False), "## VERDICT: ⚠️ ERROR"
        return

    verdict = extract_verdict(rag_response)
    color = "#00ff66" if verdict == "NORMAL" else "#ff3b3b" if verdict == "ANOMALY" else