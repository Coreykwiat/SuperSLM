import os
import sys
import json
import socket
import threading
import datetime
import ollama
import pandas as pd
import numpy as np
from docx import Document
from bs4 import BeautifulSoup
from pypdf import PdfReader
import pickle

CONFIG_FILE = "config.json"


def load_config():
    defaults = {
        "docs_directory": "./docs",
        "db_cache_file": "knowledge_base.pkl",
        "input_port": 5005,
        "output_port": 5006,
        "ollama_url": "http://127.0.0.1:11434",
        "llm_model_name": "llama3",
        "embedding_model_name": "nomic-embed-text"
    }

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = {**defaults, **json.load(f)}
    else:
        with open(CONFIG_FILE, "w") as f:
            json.dump(defaults, f, indent=4)
        data = defaults

    if not os.path.exists(data["docs_directory"]):
        os.makedirs(data["docs_directory"])
        print(f"[*] Created missing directory: {data['docs_directory']}")

    return data


cfg = load_config()
DOCS_DIRECTORY = cfg["docs_directory"]
DB_CACHE_FILE = cfg["db_cache_file"]
INPUT_PORT = cfg["input_port"]
OUTPUT_PORT = cfg["output_port"]
OLLAMA_URL = cfg["ollama_url"]
LLM_MODEL_NAME = cfg["llm_model_name"]
EMBEDDING_MODEL_NAME = cfg["embedding_model_name"]

client = ollama.Client(host=OLLAMA_URL)


def log_event(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def ensure_ollama_models():
    log_event(f"[*] Connecting to Ollama at {OLLAMA_URL}...")

    try:
        response = client.list()
        available_models = [m.model for m in response.models]

        # Check LLM model
        llm_exists = any(m.startswith(LLM_MODEL_NAME) for m in available_models)
        if not llm_exists:
            log_event(f"[!] '{LLM_MODEL_NAME}' not found. Pulling...")
            client.pull(LLM_MODEL_NAME)
            log_event(f"[*] LLM '{LLM_MODEL_NAME}' is ready.")
        else:
            log_event(f"[*] LLM '{LLM_MODEL_NAME}' is ready.")

        embed_exists = any(m.startswith(EMBEDDING_MODEL_NAME) for m in available_models)
        if not embed_exists:
            log_event(f"[!] '{EMBEDDING_MODEL_NAME}' not found. Pulling...")
            log_event("[*] This may take a few minutes on first run...")
            client.pull(EMBEDDING_MODEL_NAME)
            log_event(f"[*] Embedding model '{EMBEDDING_MODEL_NAME}' is ready.")
        else:
            log_event(f"[*] Embedding model '{EMBEDDING_MODEL_NAME}' is ready.")


    except Exception as e:
        log_event(f"\n[!] CONNECTION ERROR: {e}")
        log_event("[!] Make sure Ollama is running: 'ollama serve'")
        sys.exit(1)


def get_embedding(text):
    try:
        response = client.embeddings(
            model=EMBEDDING_MODEL_NAME,
            prompt=text
        )
        return np.array(response['embedding'])
    except Exception as e:
        log_event(f"[!] Embedding error: {e}")
        return None


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    content = ""

    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            content = "\n".join([page.extract_text() for page in reader.pages])

        elif ext == ".docx":
            doc = Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])

        elif ext in [".xlsx", ".xls"]:
            df_dict = pd.read_excel(file_path, sheet_name=None)
            sheet_texts = [f"Sheet: {name}\n{df.to_string(index=False)}"
                           for name, df in df_dict.items()]
            content = "\n\n".join(sheet_texts)

        elif ext == ".csv":
            content = pd.read_csv(file_path).to_string(index=False)

        elif ext in [".html", ".htm"]:
            # BeautifulSoup4 (US-based, FVEYS compliant)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = BeautifulSoup(f, "html.parser").get_text(separator=' ')

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

    except Exception as e:
        log_event(f" [!] Error reading {file_path}: {e}")

    return content.strip()


def reindex_docs():
    log_event(f"[*] (Re)Indexing docs in {DOCS_DIRECTORY}...")
    kb = []

    if not os.path.exists(DOCS_DIRECTORY):
        os.makedirs(DOCS_DIRECTORY)

    supported_ext = [".pdf", ".docx", ".txt", ".html", ".htm", ".xlsx", ".xls", ".csv"]
    files = [f for f in os.listdir(DOCS_DIRECTORY)
             if any(f.lower().endswith(ext) for ext in supported_ext)]

    if not files:
        log_event("[!] No supported files found in docs folder.")
        return []

    for filename in files:
        path = os.path.join(DOCS_DIRECTORY, filename)
        log_event(f" > Processing: {filename}")
        text = extract_text(path)

        if text:
            chunks = [text[i:i + 1200] for i in range(0, len(text), 1000)]

            for chunk in chunks:
                embedding = get_embedding(chunk)
                if embedding is not None:
                    kb.append({
                        "text": chunk,
                        "vec": embedding,
                        "source": filename
                    })

    with open(DB_CACHE_FILE, 'wb') as f:
        pickle.dump(kb, f)

    log_event(f"[*] Done. {len(kb)} segments indexed.")
    return kb


def needs_reindexing():
    if not os.path.exists(DB_CACHE_FILE):
        return True

    try:
        with open(DB_CACHE_FILE, 'rb') as f:
            db = pickle.load(f)

        indexed_files = set(item['source'] for item in db)
        supported_ext = [".pdf", ".docx", ".txt", ".html", ".htm", ".xlsx", ".xls", ".csv"]
        current_files = {f for f in os.listdir(DOCS_DIRECTORY)
                         if any(f.lower().endswith(ext) for ext in supported_ext)}

        return indexed_files != current_files
    except:
        return True


def load_resources():
    ensure_ollama_models()

    if needs_reindexing():
        log_event("[!] Changes detected. Re-training knowledge base...")
        db = reindex_docs()
    else:
        log_event("[*] Knowledge base up-to-date.")
        with open(DB_CACHE_FILE, 'rb') as f:
            db = pickle.load(f)

    return db


def search(db, query_text, client_ip="LOCAL"):
    if not db:
        return "Knowledge base is currently empty. Please add documents to the docs folder."

    log_event(f"[*] Processing query from {client_ip}: {query_text[:50]}...")
    q_vec = get_embedding(query_text)
    if q_vec is None:
        return "Error: Could not generate query embedding."
    scores = [(i, cosine_similarity(q_vec, item["vec"])) for i, item in enumerate(db)]
    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = min(3, len(scores))
    top_indices = [scores[i][0] for i in range(top_k)]
    context = "\n".join([db[idx]['text'] for idx in top_indices])
    response = client.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {'role': 'system', 'content': 'Answer briefly based on the provided context.'},
            {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {query_text}"}
        ]
    )
    sources = set(db[idx]['source'] for idx in top_indices)

    return f"SOURCES: {', '.join(sources)}\n\n{response.message.content}"


def socket_server():
    global knowledge_db

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    in_sock.bind(('0.0.0.0', INPUT_PORT))
    in_sock.listen(5)

    log_event(f"[*] LISTENER ACTIVE ON PORT {INPUT_PORT}")

    while True:
        conn, addr = in_sock.accept()
        client_ip = addr[0]

        try:
            data = conn.recv(8192).decode('utf-8').strip()
            if data:
                response = search(knowledge_db, data, client_ip=client_ip)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as out_sock:
                    out_sock.connect((client_ip, OUTPUT_PORT))
                    out_sock.sendall(response.encode('utf-8'))

        except Exception as e:
            log_event(f"[!] Server Error from {client_ip}: {e}")

        finally:
            conn.close()


if __name__ == "__main__":
    log_event("=" * 60)
    log_event("[i] SmartSLM Starting.....")
    log_event("[i} Created by Corey Kwiatkowski, for any issues report to https://github.com/Coreykwiat/SuperSLM")
    log_event("[i] All processing happens offline - Data Will Not Be Shared :)")
    log_event("[i] Embedding Model: nomic-embed-text (Nomic AI - US)")
    log_event("=" * 60)

    knowledge_db = load_resources()

    threading.Thread(target=socket_server, daemon=True).start()
    log_event(f"[*] Server online using {LLM_MODEL_NAME}.")

    while True:
        ui = input("\n[Local Shell] > ").strip()

        if ui.lower() in ['q', 'exit']:
            sys.exit(0)

        elif ui.lower() == 'refresh':
            knowledge_db = reindex_docs()

        elif ui.lower() == 'background':
            log_event("[*] Background Mode active. Press ENTER to return.")
            input("")
            log_event("[*] Interactive Mode restored.")

        elif ui.lower() == 'status':
            log_event(f"[*] Knowledge Base: {len(knowledge_db)} segments")
            log_event(f"[*] LLM Model: {LLM_MODEL_NAME}")
            log_event(f"[*] Embedding Model: {EMBEDDING_MODEL_NAME}")
            log_event(f"[*] SuperSLM: ✓ Active")

        elif ui:
            print("\n" + search(knowledge_db, ui, client_ip="127.0.0.1"))