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

last_query = None
last_used_indices = []
rejected_indices = set()


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

    os.makedirs(data["docs_directory"], exist_ok=True)
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


def log_event(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def ensure_ollama_models():
    try:
        models = [m.model for m in client.list().models]

        if not any(m.startswith(LLM_MODEL_NAME) for m in models):
            client.pull(LLM_MODEL_NAME)

        if not any(m.startswith(EMBEDDING_MODEL_NAME) for m in models):
            client.pull(EMBEDDING_MODEL_NAME)

    except Exception as e:
        log_event(f"[!] Ollama error: {e}")
        sys.exit(1)


def get_embedding(text):
    try:
        return np.array(
            client.embeddings(
                model=EMBEDDING_MODEL_NAME,
                prompt=text
            )["embedding"]
        )
    except Exception as e:
        log_event(f"[!] Embedding error: {e}")
        return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            return "\n".join(p.extract_text() for p in PdfReader(path).pages)
        if ext == ".docx":
            return "\n".join(p.text for p in Document(path).paragraphs)
        if ext == ".csv":
            return pd.read_csv(path).to_string(index=False)
        if ext in [".html", ".htm"]:
            with open(path, encoding="utf-8", errors="ignore") as f:
                return BeautifulSoup(f, "html.parser").get_text(" ")
        if ext == ".txt":
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        log_event(f"[!] Read error {path}: {e}")
    return ""



def reindex_docs():
    log_event("[*] Re-indexing documents...")
    kb = []

    for filename in os.listdir(DOCS_DIRECTORY):
        path = os.path.join(DOCS_DIRECTORY, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".xlsx", ".xls"):
            sheets = pd.read_excel(path, sheet_name=None)

            for sheet_name, df in sheets.items():
                df = df.fillna("")

                for row_idx, row in df.iterrows():
                    row_text = f"File: {filename}\nSheet: {sheet_name}\nRow: {row_idx}\n"
                    for col, val in row.items():
                        row_text += f"{col}: {val}\n"

                    vec = get_embedding(row_text)
                    if vec is not None:
                        kb.append({
                            "text": row_text,
                            "vec": vec,
                            "source": f"{filename}::{sheet_name}::row{row_idx}"
                        })

        elif ext in (".pdf", ".docx", ".txt", ".html", ".htm", ".csv"):
            text = extract_text(path)
            if not text:
                continue

            chunks = [text[i:i + 1200] for i in range(0, len(text), 1000)]
            for c in chunks:
                vec = get_embedding(c)
                if vec is not None:
                    kb.append({
                        "text": c,
                        "vec": vec,
                        "source": filename
                    })

    with open(DB_CACHE_FILE, "wb") as f:
        pickle.dump(kb, f)

    log_event(f"[*] Indexed {len(kb)} chunks")
    return kb


def load_resources():
    ensure_ollama_models()
    if not os.path.exists(DB_CACHE_FILE):
        return reindex_docs()
    with open(DB_CACHE_FILE, "rb") as f:
        return pickle.load(f)



def search(db, query, client_ip="LOCAL"):
    global last_query, last_used_indices

    q_vec = get_embedding(query)
    if q_vec is None:
        return "Embedding failed."

    scores = []
    for i, item in enumerate(db):
        sim = cosine_similarity(q_vec, item["vec"])
        if i in rejected_indices:
            sim *= 0.1
        scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    top_indices = []
    for i, _ in scores:
        if i not in rejected_indices:
            top_indices.append(i)
        if len(top_indices) == 3:
            break

    last_query = query
    last_used_indices = top_indices.copy()

    context = "\n".join(db[i]["text"] for i in top_indices)
    sources = ", ".join(set(db[i]["source"] for i in top_indices))

    response = client.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "Answer briefly using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    return f"SOURCES: {sources}\n\n{response.message.content}"


def socket_server():
    global knowledge_db

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", INPUT_PORT))
    s.listen(5)

    log_event(f"[*] Listening on {INPUT_PORT}")

    while True:
        conn, addr = s.accept()
        ip = addr[0]

        try:
            data = conn.recv(8192).decode().strip()

            if data.lower() == "retry":
                if last_query:
                    rejected_indices.update(last_used_indices)
                    response = search(knowledge_db, last_query, ip)
                else:
                    response = "Nothing to retry yet."
            else:
                response = search(knowledge_db, data, ip)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as out:
                out.connect((ip, OUTPUT_PORT))
                out.sendall(response.encode())

        except Exception as e:
            log_event(f"[!] Socket error: {e}")
        finally:
            conn.close()



if __name__ == "__main__":
    log_event("SuperSLM Starting")
    knowledge_db = load_resources()

    threading.Thread(target=socket_server, daemon=True).start()
    log_event("[*] Server online")
    log_event("[i] For any issue report them to [https://github.com/Coreykwiat/SuperSLM]")
    log_event("[i] All querying is done locally and is not stored by any 3rd party")
    log_event("[i] In the event of an incorrect answer send SuperSLM the command retry")

    while True:
        cmd = input("\n[Local Shell] > ").strip().lower()

        if cmd in ("q", "exit"):
            sys.exit(0)

        elif cmd in ("retry", "Retry"):
            if last_query:
                rejected_indices.update(last_used_indices)
                print(search(knowledge_db, last_query, "127.0.0.1"))
            else:
                print("Nothing to retry yet.")

        elif cmd in ("refresh", "Refresh"):
            knowledge_db = reindex_docs()

        elif cmd in ("background","Background"):
            log_event("[*] Background Mode active. Press ENTER to return.")
            input("")
            log_event("[*] Interactive Mode restored.")

        elif cmd in ("manual", "Manual"):
            print("[i] Enter 'background' to enter background mode")
            print("[i] Enter 'refresh' to reindex and create a new model")
            print("[i] Enter 'retry' to forget previous answer and requery")


        elif cmd:
            print(search(knowledge_db, cmd, "127.0.0.1"))
