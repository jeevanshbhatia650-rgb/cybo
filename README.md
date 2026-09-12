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

# 🛡️ Attack Prompt RAG

A **Retrieval-Augmented Generation (RAG)** system for network attack/anomaly analysis.

It retrieves similar historical network records from a ChromaDB vector database and provides them as context to **Qwen2.5-7B-Instruct** for classification.

### 🚀 Live Demo

https://huggingface.co/spaces/jeevrtxg/attack_prompt_rag

### 📂 Source

https://github.com/jeevanshbhatia650-rgb/cybo

---

## 🔄 Workflow

```text
Network Connection Input
          ↓
Nomic Embeddings
          ↓
ChromaDB Vector Search
          ↓
Top-5 Similar Historical Records
          ↓
   ┌──────┴──────┐
   ↓             ↓
Baseline       RAG
LLM            LLM
   │             │
   └──────┬──────┘
          ↓
   Verdict Extraction
          ↓
 NORMAL / ANOMALY
```

The system compares:

* **Baseline:** LLM prediction without retrieved context
* **RAG:** LLM prediction using historical attack records

---

## 📊 Knowledge Base

The project uses a ChromaDB collection:

```text
nslkdd_attacks
```

The persisted vector database is stored in:

```text
chroma_store_export/
```

The knowledge base contains **NSL-KDD-style network intrusion records**.

> Raw NSL-KDD CSV files are not included; the repository contains the persisted Chroma knowledge base.

---

## 🤖 Tech Stack

| Component  | Technology                       |
| ---------- | -------------------------------- |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` |
| LLM        | `Qwen/Qwen2.5-7B-Instruct`       |
| Vector DB  | ChromaDB                         |
| UI         | Gradio                           |
| Runtime    | PyTorch + Transformers           |
| Retrieval  | Top-5 similarity search          |

---

## 🧪 Sample Inputs

### Normal

```text
protocol_type=tcp, service=http, flag=SF, src_bytes=1200, dst_bytes=4500, count=1
```

### Suspicious

```text
protocol_type=tcp, service=http, flag=S0, src_bytes=0, dst_bytes=0, count=40, serror_rate=0.95
```

### Port-scan-like

```text
protocol_type=tcp, service=private, flag=S0, src_bytes=0, dst_bytes=0, count=80, diff_srv_rate=0.95
```

> These are illustrative test inputs, not guaranteed ground-truth labels.

---

## 🖥️ UI

The Gradio interface provides:

* 🔎 Retrieval logs
* 📚 Top-5 retrieved records
* 🤖 Baseline LLM response
* 🧠 RAG response
* 🚨 Final anomaly verdict

---

## 📁 Structure

```text
cybo/
├── app.py
├── requirements.txt
├── chroma_store_export/
└── README.md
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/jeevanshbhatia650-rgb/cybo.git
cd cybo

python -m venv .venv
# Activate the environment

pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

---

## 📈 RAG vs Baseline

The project can be used to evaluate whether retrieved historical examples improve LLM-based network anomaly classification.

Recommended metrics:

```text
Accuracy
Precision
Recall
F1 Score
False Positive Rate
False Negative Rate
```

---

## ⚠️ Limitations

* Prototype-level IDS/RAG system
* Simple regex-based verdict extraction
* No calibrated confidence score
* 7B LLM can be resource-intensive
* Requires proper held-out evaluation for meaningful performance claims

---

## 🚀 Future Work

* Real-time network traffic ingestion
* Hybrid vector + keyword retrieval
* Retrieval reranking
* Structured JSON outputs
* Confidence scoring
* Adversarial robustness testing
* RAG vs baseline benchmarking

---

## 👤 Author

**Jeevansh Bhatia**

GitHub:
https://github.com/jeevanshbhatia650-rgb
