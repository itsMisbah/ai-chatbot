# BizAssist AI — PDF-Powered Business Chatbot

An intelligent customer support chatbot that lets businesses upload any PDF (menu, catalog, policy doc, FAQ) and instantly get an AI assistant that answers customer questions accurately — powered by RAG (Retrieval-Augmented Generation).

---

## How It Works (Architecture)

```
User Question
     │
     ▼
[HuggingFace Embeddings]        ← Converts question to vector
     │
     ▼
[FAISS Vector Store]            ← Searches for relevant PDF chunks
     │
     ▼
[MultiQuery Retriever]          ← Rephrases query 3 ways for better recall
     │
     ▼
[Top K Relevant Chunks]         ← Retrieved context from the PDF
     │
     ▼
[Groq LLM - LLaMA 3.1 8B]       ← Generates answer from context only
     │
     ▼
[Streamlit Chat UI]             ← Displays response to user
```
---

## ⚙️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **UI** | Streamlit | Fast to build, chat-native |
| **LLM** | Groq + LLaMA 3.1 8B | Free tier, extremely fast inference |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Free, lightweight, accurate |
| **Vector Store** | FAISS (via LangChain) | In-memory, no DB setup needed |
| **PDF Loader** | LangChain PyPDFLoader | Reliable text extraction |
| **Retriever** | MultiQueryRetriever | Better recall than single-query search |
| **Orchestration** | LangChain | Chains, retrievers, prompt management |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/bizassist-ai.git
cd bizassist-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Add your PDF

Place your business PDF in the root directory and update this line in `chatbot.py`:

```python
pdf_name = "./your_business_file.pdf"
```

### 5. Run the app

```bash
streamlit run chatbot.py
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-groq
langchain-huggingface
langchain-community
pypdf
python-dotenv
faiss-cpu
sentence-transformers
```
---

## 🙋 About

Built by **[Misbah Shahzadi]** as part of an exploration into practical LLM applications for small businesses.

- 🔗 [LinkedIn](https://linkedin.com/in/misbah-shahzadi)
- 🐙 [GitHub](https://github.com/itsMisbah)
