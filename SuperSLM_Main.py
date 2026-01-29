import os
import sys
import json
import torch
import pickle
import socket
import threading
import datetime
import ollama
import pandas as pd
from docx import Document
from bs4 import BeautifulSoup
from pypdf import PdfReader

CONFIG_FILE = "config.json"


def load_config():
    defaults = {
        "docs_directory": "./docs",
        "model_save_path": "./local_model",
        "db_cache_file": "knowledge_base.pkl",
        "input_port": 5005,
        "output_port": 5006,
        "ollama_url": "http://127.0.0.1:11434",
        "embed_model_name": "nomic-embed-text",  # CHANGE: FVEYS compliant embedding
        "llm_model_name": "llama3"
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
MODEL_SAVE_PATH = cfg["model_save_path"]
DB_CACHE_FILE = cfg["db_cache_file"]
INPUT_PORT = cfg["input_port"]
OUTPUT_PORT = cfg["output_port"]
OLLAMA_URL = cfg["ollama_url"]
EMBED_MODEL_NAME = cfg["embed_model_name"]
LLM_MODEL_NAME = cfg["llm_model_name"]

client = ollama.Client(host=OLLAMA_URL)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def log_event(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def ensure_ollama_model():
    log_event(f"[*] Connecting to Ollama at {OLLAMA_URL}...")
    try:
        response = client.list()
        model_exists = any(m.model.startswith(LLM_MODEL_NAME) for m in response.models)
        if not model_exists:
            log_event(f"[!] '{LLM_MODEL_NAME}' not found. Pulling...")
            client.pull(LLM_MODEL_NAME)
            log_event(f"[*] Brain '{LLM_MODEL_NAME}' is ready.")
        embed_exists = any(m.model.startswith(EMBED_MODEL_NAME) for m in response.models)
        if not embed_exists:
            log_event(f"[!] '{EMBED_MODEL_NAME}' not found. Pulling...")
            client.pull(EMBED_MODEL_NAME)
            log_event(f"[*] Embedding model '{EMBED_MODEL_NAME}' is ready.")
    except Exception as e:
        log_event(f"\n[!] CONNECTION ERROR: {e}")
        sys.exit(1)


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    content = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            content = "\n".join([page.extract_text() or "" for page in reader.pages])
        elif ext == ".docx":
            doc = Document(file_path)
            parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text)

            for table in doc.tables:
                table_text = "\n[TABLE START]\n"
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_text += " | ".join(row_data) + "\n"
                table_text += "[TABLE END]\n"
                parts.append(table_text)

            content = "\n".join(parts)

        elif ext in [".xlsx", ".xls"]:
            df_dict = pd.read_excel(file_path, sheet_name=None)
            sheet_texts = []
            for name, df in df_dict.items():
                sheet_text = f"\n[TABLE: Sheet {name}]\n"
                sheet_text += df.to_string(index=False)
                sheet_text += "\n[TABLE END]\n"
                sheet_texts.append(sheet_text)
            content = "\n\n".join(sheet_texts)

        elif ext == ".csv":
            df = pd.read_csv(file_path)
            content = "[TABLE START]\n"
            content += df.to_string(index=False)
            content += "\n[TABLE END]"

        elif ext in [".html", ".htm"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
                parts = []
                for table in soup.find_all('table'):
                    table_text = "\n[TABLE START]\n"
                    for row in table.find_all('tr'):
                        cells = row.find_all(['td', 'th'])
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        table_text += " | ".join(row_data) + "\n"
                    table_text += "[TABLE END]\n"
                    parts.append(table_text)
                    # Remove table from soup so we don't duplicate in text
                    table.decompose()
                text_content = soup.get_text(separator=' ')
                if text_content.strip():
                    parts.insert(0, text_content)

                content = "\n\n".join(parts)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
    except Exception as e:
        log_event(f" [!] Error reading {file_path}: {e}")
    return content.strip()


def reindex_docs(model):
    log_event(f"[*] (Re)Indexing docs in {DOCS_DIRECTORY}...")
    kb = []
    if not os.path.exists(DOCS_DIRECTORY):
        os.makedirs(DOCS_DIRECTORY)
    supported_ext = [".pdf", ".docx", ".txt", ".html", ".htm", ".xlsx", ".xls", ".csv"]
    files = [f for f in os.listdir(DOCS_DIRECTORY) if any(f.lower().endswith(ext) for ext in supported_ext)]
    if not files:
        log_event("[!] No supported files found in docs folder.")
        return []
    for filename in files:
        path = os.path.join(DOCS_DIRECTORY, filename)
        log_event(f" > Processing: {filename}")
        text = extract_text(path)
        if text:
            chunks = smart_chunk_with_tables(text)
            for chunk in chunks:
                vec = model.encode(chunk)
                kb.append({"text": chunk, "vec": vec, "source": filename})
    with open(DB_CACHE_FILE, 'wb') as f:
        pickle.dump(kb, f)
    log_event(f"[*] Done. {len(kb)} segments indexed.")
    return kb


def smart_chunk_with_tables(text, max_chunk_size=1200, step_size=1000):
    chunks = []
    parts = []
    current_pos = 0

    while current_pos < len(text):
        table_start = text.find('[TABLE', current_pos)

        if table_start == -1:
            remaining = text[current_pos:]
            if remaining.strip():
                parts.append(('text', remaining))
            break
        if table_start > current_pos:
            before_table = text[current_pos:table_start]
            if before_table.strip():
                parts.append(('text', before_table))
        table_end = text.find('[TABLE END]', table_start)
        if table_end == -1:
            parts.append(('text', text[table_start:]))
            break
        table_end += len('[TABLE END]')
        table_content = text[table_start:table_end]
        parts.append(('table', table_content))

        current_pos = table_end
    current_chunk = ""

    for part_type, content in parts:
        if part_type == 'table':
            if len(content) > max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(content)
            else:
                if len(current_chunk) + len(content) > max_chunk_size and current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = content
                else:
                    current_chunk += "\n" + content
        else:
            text_chunks = [content[i:i + max_chunk_size] for i in range(0, len(content), step_size)]

            for text_chunk in text_chunks:
                if len(current_chunk) + len(text_chunk) > max_chunk_size and current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = text_chunk
                else:
                    current_chunk += "\n" + text_chunk

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def needs_reindexing():
    if not os.path.exists(DB_CACHE_FILE):
        return True
    try:
        with open(DB_CACHE_FILE, 'rb') as f:
            db = pickle.load(f)
        indexed_files = set(item['source'] for item in db)
        current_files = {f for f in os.listdir(DOCS_DIRECTORY) if any(
            f.lower().endswith(ext) for ext in [".pdf", ".docx", ".txt", ".html", ".htm", ".xlsx", ".xls", ".csv"])}
        return indexed_files != current_files
    except:
        return True


class OllamaEmbeddingModel:
    def __init__(self, model_name, client):
        self.model_name = model_name
        self.client = client

    def encode(self, text, convert_to_tensor=True):
        if isinstance(text, list):
            embeddings = []
            for t in text:
                response = self.client.embeddings(model=self.model_name, prompt=t)
                embeddings.append(response['embedding'])
            if convert_to_tensor:
                return torch.tensor(embeddings)
            return embeddings
        else:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            if convert_to_tensor:
                return torch.tensor(response['embedding'])
            return response['embedding']

    def save(self, path):
        pass


def load_resources():
    ensure_ollama_model()
    model = OllamaEmbeddingModel(EMBED_MODEL_NAME, client)

    if needs_reindexing():
        log_event("[!] Changes detected. Re-training knowledge base...")
        db = reindex_docs(model)
    else:
        log_event("[*] Knowledge base up-to-date.")
        with open(DB_CACHE_FILE, 'rb') as f:
            db = pickle.load(f)
    return model, db


def search(model, db, query_text, client_ip="LOCAL"):
    if not db:
        return "Knowledge base is currently empty. Please add documents to the docs folder."
    log_event(f"[*] Processing query from {client_ip}: {query_text[:50]}...")

    q_vec = model.encode(query_text, convert_to_tensor=True)
    corpus_embeddings = torch.stack(
        [torch.tensor(item["vec"]) if not isinstance(item["vec"], torch.Tensor) else item["vec"] for item in db])

    from torch.nn.functional import cosine_similarity as torch_cosine_sim
    scores = torch_cosine_sim(q_vec.unsqueeze(0), corpus_embeddings, dim=1)
    top_results = torch.topk(scores, k=min(3, len(db)))

    context_parts = []
    for idx in top_results.indices:
        chunk_text = db[int(idx)]['text']
        source = db[int(idx)]['source']

        if '[TABLE' in chunk_text:
            context_parts.append(f"From {source}:\n{chunk_text}")
        else:
            context_parts.append(chunk_text)

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """Answer briefly based on the provided context. 
If the context contains tables (marked with [TABLE START] and [TABLE END]), carefully analyze the relationships between columns and rows.
When referencing table data, maintain the associations between related fields."""

    response = client.chat(model=LLM_MODEL_NAME, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Context: {context}\n\nQuestion: {query_text}"}
    ])
    sources = set(db[int(idx)]['source'] for idx in top_results.indices)
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
                response = search(embed_model, knowledge_db, data, client_ip=client_ip)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as out_sock:
                    out_sock.connect((client_ip, OUTPUT_PORT))
                    out_sock.sendall(response.encode('utf-8'))
        except Exception as e:
            log_event(f"[!] Server Error from {client_ip}: {e}")
        finally:
            conn.close()


if __name__ == "__main__":
    embed_model, knowledge_db = load_resources()
    threading.Thread(target=socket_server, daemon=True).start()
    log_event(f"[*] Server online using {LLM_MODEL_NAME}.")
    while True:
        ui = input("\n[Local Shell] > ").strip()
        if ui.lower() in ['q', 'exit']:
            sys.exit(0)
        elif ui.lower() == 'refresh':
            knowledge_db = reindex_docs(embed_model)
        elif ui.lower() == 'background':
            log_event("[*] Background Mode active. Press ENTER to return.")
            input("")
            log_event("[*] Interactive Mode restored.")
        elif ui:
            print("\n" + search(embed_model, knowledge_db, ui, client_ip="127.0.0.1"))