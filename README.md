# MultiDocChat

MultiDocChat is a Streamlit-based Retrieval-Augmented Generation (RAG) app that lets you upload documents and ask questions about them using Google Gemini and FAISS vector search.

It supports a wide mix of file types, extracts text from them, chunks the content, embeds it with Gemini, and then answers questions based only on the uploaded material.

## Features

- Upload multiple files at once
- Supports PDFs, spreadsheets, Word documents, PowerPoint slides, plain text, source code, SQL databases, and more
- Uses Gemini embeddings + FAISS for semantic retrieval
- Answers questions grounded only in the context of the uploaded files
- Keeps the vector index in memory for the current session
- Shows source files used for each answer

## Supported file types

The app supports:

- PDF: `.pdf`
- Spreadsheets: `.csv`, `.xlsx`
- Documents: `.docx`, `.txt`, `.md`, `.rst`, `.html`, `.xml`
- Presentations: `.pptx`
- Databases: `.db`, `.sqlite`, `.sqlite3`
- Code and config files: `.py`, `.js`, `.ts`, `.java`, `.php`, `.sql`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, and more

## Tech stack

- Streamlit
- LangChain
- LangChain Google GenAI
- FAISS
- Google Generative AI
- PyPDF2
- pandas
- python-docx
- python-pptx

## Project structure

```text
MultiDocChat/
├── app.py                # Streamlit application
├── file_loaders.py       # File extraction and text normalization
├── requirements.txt      # Python dependencies
├── .streamlit/           # Streamlit config and secrets (optional)
├── files/                # Sample or temporary upload storage
├── faiss_index/         # Any persisted FAISS artifacts (if used in local experiments)
├── MultiFile_Reader.ipynb  # Notebook prototype or exploratory work
└── README.md            # Project documentation
```

## Prerequisites

- Python 3.9+
- A Google Gemini API key

## Installation

1. Clone or download the project.
2. Open a terminal in the project folder.
3. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Set your Gemini API key before running the app.

### Option 1: Environment variable

Linux/macOS:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

### Option 2: Streamlit secrets

Create a file named `.streamlit/secrets.toml` in the project root:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

The app will automatically read this key when running in Streamlit.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## How it works

1. You upload one or more files.
2. The app reads the content using the loaders in `file_loaders.py`.
3. Text is split into chunks.
4. Each chunk is embedded using Gemini embeddings.
5. FAISS stores the vectors in memory.
6. When you ask a question, the app retrieves the most relevant chunks.
7. Gemini answers using only the retrieved document context.

## Usage example

1. Launch the app.
2. Upload files such as a PDF, spreadsheet, or source code project.
3. Click `Process files`.
4. Ask a question like:
   - "What are the key risks mentioned in this report?"
   - "Summarize the sales data by region."
   - "Which files mention authentication?"

The app will return a grounded answer and list the relevant source files used.

## Notes

- The app is designed for local/demo usage and keeps the vector store in memory.
- PDFs without a usable text layer may not be extractable.
- Large tables and databases are truncated to a maximum row limit for indexing efficiency.
- Answers are intentionally constrained to the uploaded content and do not rely on outside knowledge.
