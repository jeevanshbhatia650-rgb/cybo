

# 🛡️ Attack Prompt RAG

A **RAG-based network anomaly detection system** that retrieves similar historical attack records and uses **Qwen2.5-7B-Instruct** to classify new network connections.

## 🚀 Live Demo

https://huggingface.co/spaces/jeevrtxg/attack_prompt_rag

## 🔄 Workflow

```text
Network Connection
       ↓
Nomic Embeddings
       ↓
ChromaDB Vector Search
       ↓
Top-5 Similar Records
       ↓
 ┌─────┴─────┐
 ↓           ↓
Baseline     RAG
LLM          LLM
 ↓           ↓
 └─────┬─────┘
       ↓
Verdict Extraction
       ↓
NORMAL / ANOMALY
```

### Baseline vs RAG

* **Baseline:** LLM classification without historical context
* **RAG:** LLM classification using retrieved historical records

## 📊 Knowledge Base

ChromaDB collection:

```text
nslkdd_attacks
```

Stored in:

```text
chroma_store_export/
```

The knowledge base contains **NSL-KDD-style network intrusion records**.

> Raw NSL-KDD CSV files are not included. The persisted ChromaDB knowledge base is included.

## 🤖 Tech Stack

| Component  | Technology                       |
| ---------- | -------------------------------- |
| Embeddings | `nomic-ai/nomic-embed-text-v1.5` |
| LLM        | `Qwen/Qwen2.5-7B-Instruct`       |
| Vector DB  | ChromaDB                         |
| UI         | Gradio                           |
| Runtime    | PyTorch + Transformers           |

## 🧪 Sample Inputs

### Normal

```text
protocol_type=tcp, service=http, flag=SF, src_bytes=1200, dst_bytes=4500, count=1
```

### Suspicious

```text
protocol_type=tcp, service=http, flag=S0, src_bytes=0, dst_bytes=0, count=40, serror_rate=0.95
```

### Port Scan

```text
protocol_type=tcp, service=private, flag=S0, src_bytes=0, dst_bytes=0, count=80, diff_srv_rate=0.95
```

> These are illustrative test inputs and are not guaranteed ground-truth labels.

## 🖥️ UI

The application displays:

* Retrieval logs
* Top-5 similar records
* Baseline LLM response
* RAG response
* Final anomaly verdict

## 📁 Project Structure

```text
cybo/
├── app.py
├── requirements.txt
├── chroma_store_export/
└── README.md
```

## ⚙️ Run Locally

```bash
git clone https://github.com/jeevanshbhatia650-rgb/cybo.git
cd cybo

python -m venv .venv
pip install -r requirements.txt

python app.py
```

Open:

```text
http://127.0.0.1:7860
```

## 📈 Evaluation

The system can be evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* False Positive Rate
* False Negative Rate
* RAG vs Baseline improvement

## 🚀 Future Work

* Real-time network traffic ingestion
* Hybrid retrieval
* Retrieval reranking
* Structured JSON output
* Confidence scoring
* Adversarial robustness testing
* RAG vs baseline benchmarking

## 👤 Author

**Jeevansh Bhatia**

GitHub: https://github.com/jeevanshbhatia650-rgb
