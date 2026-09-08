# 🧑‍💼 HR Policy Assistant — RAG

An AI-powered HR Policy Assistant built with **Retrieval-Augmented Generation (RAG)**.

Users can upload one or more HR policy PDFs and ask natural-language questions. The application extracts the policy text, creates semantic embeddings, searches the most relevant sections using FAISS, and generates a grounded answer using Groq's `openai/gpt-oss-20b` model.

## 🚀 Live Demo

Add your Streamlit Cloud URL here after deployment.

## ✨ Features

* 📄 Upload multiple HR policy PDFs
* 🔍 Semantic search over uploaded policies
* 🧠 Sentence Transformers embeddings
* ⚡ FAISS vector similarity search
* 🤖 Groq `openai/gpt-oss-20b`
* 📚 Source document and page references
* 📊 Retrieval relevance scores
* 💬 Conversational question answering
* 🛡️ Grounded responses to reduce hallucinations
* 🗑️ Clear and rebuild the knowledge base
* 🌐 Deployable through Streamlit Community Cloud

## 🏗️ Architecture

```text
                HR Policy PDFs
                       │
                       ▼
                  PyMuPDF
                       │
                       ▼
                Text Extraction
                       │
                       ▼
                 Text Chunking
                       │
                       ▼
          Sentence Transformers
                       │
                       ▼
                 FAISS Index
                       │
                       │
                User Question
                       │
                       ▼
               Query Embedding
                       │
                       ▼
              Similarity Search
                       │
                       ▼
             Relevant Policy Chunks
                       │
                       ▼
             Groq openai/gpt-oss-20b
                       │
                       ▼
             Grounded HR Answer
                       │
                       ▼
             Sources + Page Numbers
```

## 🛠️ Tech Stack

| Technology                | Purpose                  |
| ------------------------- | ------------------------ |
| Python                    | Application development  |
| Streamlit                 | Web interface            |
| PyMuPDF                   | PDF text extraction      |
| Sentence Transformers     | Text embeddings          |
| FAISS                     | Vector similarity search |
| NumPy                     | Vector processing        |
| Groq                      | LLM inference            |
| `openai/gpt-oss-20b`      | Language model           |
| GitHub                    | Source control           |
| Streamlit Community Cloud | Deployment               |

## 📁 Project Structure

```text
hr-policy-assistant-rag/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔐 API Key

The application requires a Groq API key.

The key should **never** be committed to GitHub.

For Streamlit Community Cloud, add the following secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## 💡 Example Questions

After uploading HR policy documents, try questions such as:

```text
How many annual leave days are employees entitled to?

What is the company's remote work policy?

How much sick leave is available?

How far in advance should I request annual leave?

Can unused leave be carried forward?

What should an employee do if they cannot attend work?

What are the rules regarding workplace harassment?
```

## 🧠 RAG Pipeline

The application follows four main stages:

### 1. Retrieval

The user's question is converted into an embedding using Sentence Transformers.

### 2. Similarity Search

FAISS searches the vector index for the most relevant policy chunks.

### 3. Augmentation

The retrieved policy sections are inserted into the LLM prompt as context.

### 4. Generation

Groq's `openai/gpt-oss-20b` generates an answer using the retrieved policy context.

## 🛡️ Hallucination Control

The assistant is instructed to:

* Use only the uploaded policy context.
* Avoid inventing policies.
* Avoid using outside knowledge.
* Say when information cannot be found.
* Reference source documents and pages.
* Recommend contacting HR when policies are ambiguous.

## ⚠️ Privacy Note

Do not upload confidential employee records, personally identifiable information, or sensitive company documents unless you have authorization to process them through the application and its hosting/AI providers.

## 🚀 Deployment

The application can be deployed using Streamlit Community Cloud.

1. Create a GitHub repository.
2. Add `app.py`.
3. Add `requirements.txt`.
4. Add `.gitignore`.
5. Add this README.
6. Connect GitHub to Streamlit Community Cloud.
7. Select the repository and `app.py`.
8. Add `GROQ_API_KEY` through Streamlit secrets.
9. Deploy.

## 🔮 Future Improvements

Possible upgrades include:

* Hybrid search
* Reranking
* Multi-user authentication
* Persistent vector databases
* OCR for scanned PDFs
* Document management
* Admin dashboard
* Conversation memory
* Feedback collection
* Answer confidence estimation
* Policy comparison
* Multi-language HR support

## 👩‍💻 Author

**Amna Kaleem**

Built as a Generative AI / RAG portfolio project.
