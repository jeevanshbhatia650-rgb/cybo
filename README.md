---

title: Attack Prompt RAG
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
python_version: '3.13'
app_file: app.py
pinned: false
-------------

# Attack Prompt RAG

A retrieval-augmented network anomaly analysis system that compares a new network connection against historically labelled attack records and asks an instruction-tuned LLM to classify it as **NORMAL** or **ANOMALY**.

The project uses a Gradio security-console UI, Chroma vector retrieval, `nomic-ai/nomic-embed-text-v1.5` embeddings, and `Qwen/Qwen2.5-7B-Instruct`.

> **Live demo:** https://huggingface.co/spaces/jeevrtxg/attack_prompt_rag
>
> **Source:** https://github.com/jeevanshbhatia650-rgb/cybo

---

## ✨ What it does

For every new network-connection description the application runs two paths:

1. **Baseline LLM** — classification without retrieved context.
2. **RAG LLM** — retrieves the five most similar historical records and classifies the new connection using them as context.

The UI exposes retrieval logs, baseline output, RAG output, and a final verdict:

* `NORMAL`
* `ANOMALY DETECTED`
* `UNCLEAR`
* `PIPELINE ERROR`

---

## 🧠 System Workflow

```text
New Network Connection / Test Prompt
                │
                ▼
┌──────────────────────────────────┐
│ Nomic Embedding Model            │
│ nomic-ai/nomic-embed-text-v1.5   │
└────────────────┬─────────────────┘
                 │
                 ▼
┌──────────────────────────────────┐
│ Chroma Vector DB                 │
│ collection: nslkdd_attacks       │
│ historical labelled records      │
└────────────────┬─────────────────┘
                 │ Top-5 retrieval
          ┌──────┴──────┐
          ▼             ▼
┌────────────────┐ ┌────────────────────┐
│ Baseline LLM   │ │ RAG LLM            │
│ no context     │ │ + retrieved context│
└───────┬────────┘ └─────────┬──────────┘
        └──────────┬─────────┘
                   ▼
          Verdict Extraction
                   │
          NORMAL / ANOMALY
```

### Request pipeline

```text
Input
  ↓
Embedding
  ↓
Chroma similarity search
  ↓
Top-5 historical records
  ↓
┌──────────────────────┬─────────────────────────┐
│ Baseline prompt      │ RAG prompt              │
│ No historical context│ Historical context      │
└──────────┬───────────┴────────────┬────────────┘
           ↓                        ↓
        Qwen 2.5 7B Instruct
                    ↓
          Generated explanation
                    ↓
          Verdict extraction
                    ↓
       NORMAL / ANOMALY / UNCLEAR
```

---

## 📊 Dataset / Knowledge Base

The Chroma collection used by `app.py` is:

```text
nslkdd_attacks
```

The historical knowledge base is **NSL-KDD-style network intrusion data** stored as embedded records in:

```text
chroma_store_export/
```

Each retrieved item is used with:

* historical network-connection text
* classification label stored in Chroma metadata
* similarity distance

The RAG prompt presents these records as historical evidence.

### Dataset note

The repository contains the **Chroma persistence/export** rather than the original raw NSL-KDD CSV files.

Therefore, the application queries the embedded historical knowledge base at runtime rather than training a classifier from raw CSV files.

---

## 🔎 Retrieval Strategy

For every new query:

1. Encode the input using `nomic-ai/nomic-embed-text-v1.5`.
2. Query the Chroma collection `nslkdd_attacks`.
3. Retrieve the **Top-5** most similar records.
4. Include each record's:

   * label
   * distance
   * text
5. Pass the retrieved context to Qwen 2.5 7B Instruct.
6. Generate an explanation and classification.

The implementation uses Chroma's:

```python
documents
metadatas
distances
```

to construct the retrieval trace.

---

## 🤖 Models & Technology Stack

| Component       | Technology                         |
| --------------- | ---------------------------------- |
| Embeddings      | `nomic-ai/nomic-embed-text-v1.5`   |
| Generator       | `Qwen/Qwen2.5-7B-Instruct`         |
| Vector Database | ChromaDB                           |
| UI              | Gradio 4.44.1                      |
| ML Runtime      | PyTorch + Transformers             |
| Retrieval       | Chroma similarity search           |
| Device          | CUDA when available, otherwise CPU |

The application automatically selects the execution device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

The actual model names and device-selection logic are implemented in `app.py`.

---

## 🧪 Sample Inputs

The application accepts a **free-form network connection description**.

### Test 1 — Normal HTTP traffic

```text
protocol_type=tcp, service=http, flag=SF, src_bytes=1200, dst_bytes=4500, count=1, same_srv_rate=1.0, dst_host_srv_count=20
```

Expected direction:

```text
NORMAL
```

---

### Test 2 — Suspicious connection burst

```text
protocol_type=tcp, service=http, flag=S0, src_bytes=0, dst_bytes=0, count=40, srv_count=38, serror_rate=0.95, srv_serror_rate=0.97
```

Expected direction:

```text
ANOMALY
```

---

### Test 3 — Port-scan-like behaviour

```text
protocol_type=tcp, service=private, flag=S0, src_bytes=0, dst_bytes=0, count=80, srv_count=75, same_srv_rate=0.05, diff_srv_rate=0.95
```

Expected direction:

```text
ANOMALY
```

---

### Test 4 — Natural-language connection

```text
A TCP HTTP connection transferred 1500 bytes from the source and 5000 bytes to the destination, completed successfully, and showed no repeated connection errors.
```

Useful for comparing:

```text
Baseline LLM
      vs
RAG LLM
```

> Expected labels above are testing hypotheses, not ground-truth guarantees. This is an LLM/RAG research prototype, not a production IDS.

---

## 🖥️ UI / Observability

The interface is designed as a security-analysis console.

### 1. Retrieval Terminal

Displays:

```text
[INIT] embedding query...
[OK] embedding complete
[QUERY] searching vector store (chroma)...

#1 [label] dist=...
#2 [label] dist=...
#3 [label] dist=...
#4 [label] dist=...
#5 [label] dist=...

[DONE] retrieval complete
```

This allows users to inspect which historical records were retrieved.

### 2. Baseline Terminal

Runs:

```text
Qwen2.5-7B-Instruct
```

without historical context.

### 3. RAG Terminal

Runs the same LLM with retrieved historical records supplied as context.

### 4. Verdict Bar

The generated RAG response is scanned for:

```text
normal
anomaly
```

and mapped to the corresponding UI state.

---

## 📁 Repository Structure

```text
cybo/
│
├── app.py
│   └── Gradio UI + complete RAG inference pipeline
│
├── requirements.txt
│   └── Python dependencies
│
├── chroma_store_export/
│   ├── chroma.sqlite3
│   └── <Chroma collection data>
│
├── .gitattributes
│
└── README.md
```

The current repository contains exactly these main project components.

---

## ⚙️ Installation

### 1. Clone

```bash
git clone https://github.com/jeevanshbhatia650-rgb/cybo.git
cd cybo
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The repository currently specifies Gradio 4.44.1, ChromaDB, Sentence Transformers, Transformers, PyTorch, Accelerate and supporting packages.

---

## ▶️ Run Locally

```bash
python app.py
```

The application configures Gradio to use:

```text
http://127.0.0.1:7860
```

On first execution, Hugging Face model files may need to be downloaded:

```text
nomic-ai/nomic-embed-text-v1.5
Qwen/Qwen2.5-7B-Instruct
```

A CUDA-capable GPU is recommended for practical inference speed and memory usage.

---

## 🌐 Live Test

### Hugging Face Space

**https://huggingface.co/spaces/jeevrtxg/attack_prompt_rag**

Recommended test workflow:

```text
Open Space
   ↓
Paste sample network connection
   ↓
Run analysis
   ↓
Inspect retrieval terminal
   ↓
Inspect Top-5 records
   ↓
Compare Baseline LLM
   ↓
Compare RAG LLM
   ↓
Check final verdict
```

---

## 🔬 RAG vs Baseline Experiment

One of the main experimental purposes of this project is to compare LLM reasoning with and without retrieval.

| Experiment | Context                  | Purpose                                                 |
| ---------- | ------------------------ | ------------------------------------------------------- |
| Baseline   | None                     | Measure LLM-only reasoning                              |
| RAG        | Top-5 historical records | Test contextual classification using retrieved examples |

For a rigorous evaluation, use a held-out labelled dataset and measure:

* Accuracy
* Precision
* Recall
* F1-score
* False-positive rate
* False-negative rate
* RAG vs baseline improvement
* Retrieval distance vs classification correctness

---

## 🛠️ Core Implementation

The main inference functions are:

```python
retrieve_similar_attacks(query_text, top_k=5)

build_rag_prompt(query_text, retrieved_results)

build_baseline_prompt(query_text)

ask_llm(prompt)

extract_verdict(text)
```

### Core RAG operation

```text
                  User Query
                      │
                      ▼
                Text Embedding
                      │
                      ▼
               ChromaDB Search
                      │
                      ▼
             Top-5 Similar Records
                      │
                      ▼
              Context Construction
                      │
                      ▼
             Qwen2.5-7B-Instruct
                      │
                      ▼
               Generated Analysis
                      │
                      ▼
             Verdict Extraction
                      │
              ┌───────┴───────┐
              ▼               ▼
           NORMAL          ANOMALY
```

---

## 📦 Dependencies

The current project dependency file includes:

```text
gradio==4.44.1
starlette==0.37.2
fastapi==0.110.3
audioop-lts
huggingface_hub==0.23.4
chromadb
sentence-transformers
transformers
torch
accelerate
einops
```

---

## ⚠️ Limitations

* This is an **LLM/RAG security research prototype**, not a production intrusion-detection system.
* Final verdict extraction currently relies on a simple regex.
* No calibrated probability/confidence score is produced.
* Raw NSL-KDD CSV files are not included.
* CPU inference with a 7B model can be slow and memory-intensive.
* A single prompt cannot establish IDS performance.
* Retrieved examples can influence LLM reasoning, so retrieval quality directly affects the final analysis.

---

## 🚀 Future Improvements

### Evaluation

* [ ] Held-out NSL-KDD evaluation
* [ ] Confusion matrix
* [ ] Precision / Recall / F1
* [ ] False-positive / false-negative analysis
* [ ] RAG vs baseline benchmark

### Retrieval

* [ ] Retrieval threshold
* [ ] Hybrid keyword + vector search
* [ ] Multiple embedding models
* [ ] Retrieval reranking
* [ ] Retrieval-quality metrics

### LLM

* [ ] Structured JSON output
* [ ] Confidence scoring
* [ ] Quantized models
* [ ] Multiple LLM comparison
* [ ] Explainable evidence extraction

### Security

* [ ] Adversarial traffic examples
* [ ] Robustness testing
* [ ] Out-of-distribution detection
* [ ] Real-time network-stream ingestion
* [ ] Production IDS integration

---

## 📜 License

No explicit repository license is currently specified.

Add a `LICENSE` file before distributing or reusing the project under a formal open-source license.

---

## 👤 Author

**Jeevansh Bhatia**

GitHub:

https://github.com/jeevanshbhatia650-rgb

Project:

https://github.com/jeevanshbhatia650-rgb/cybo

Live Demo:

https://huggingface.co/spaces/jeevrtxg/attack_prompt_rag
