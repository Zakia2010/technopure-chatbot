import os, re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from openai import OpenAI

# ---------- Réglages ----------
BASE = "https://www.technopure.ma/"
ALLOWED_NETLOC = urlparse(BASE).netloc.replace("www.", "")

app = FastAPI(title="Technopure Site-only Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.technopure.ma", "https://technopure.ma", "*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- OpenAI + Chroma ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("technopure")

# Utilisation des embeddings OpenAI (léger et cloud)
def embed_texts(texts):
    """Crée les embeddings via l'API OpenAI (pas de modèle local)"""
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [r.embedding for r in res.data]

# ---------- Extraction du contenu du site ----------
def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","header","footer","form","nav","aside"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text

def chunk(text: str, max_chars=800):
    parts = []
    while len(text) > max_chars:
        cut = text.rfind(". ", 0, max_chars)
        cut = cut if cut != -1 else max_chars
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts

def crawl(start=BASE, limit=300):
    seen, queue, pages = set(), [start], []
    while queue and len(seen) < limit:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
            text = clean_text(r.text)
            if len(text) < 200:
                continue
            pages.append((url, text))
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                u = urljoin(url, a["href"])
                p = urlparse(u)
                host = p.netloc.replace("www.", "")
                if host == ALLOWED_NETLOC and p.scheme in ("http", "https"):
                    u = u.split("#")[0]
                    if u.endswith((".pdf", ".zip", ".jpg", ".png", ".jpeg", ".gif", ".webp")):
                        continue
                    queue.append(u)
        except Exception:
            continue
    return pages

# ---------- Construction de l'index ----------
def build_index():
    pages = crawl()
    docs, metas = [], []
    for url, text in pages:
        for part in chunk(text):
            docs.append(part)
            metas.append({"url": url})
    if not docs:
        return
    embeddings = embed_texts(docs)
    ids = [str(i) for i in range(len(docs))]
    collection.add(documents=docs, metadatas=metas, embeddings=embeddings, ids=ids)
    print(f"✅ {len(docs)} passages indexés depuis technopure.ma")

build_index()

SYSTEM_PROMPT = (
    "Tu es l'assistant Technopure. Réponds UNIQUEMENT avec les informations "
    "du contexte (extraits de technopure.ma). Sinon dis que tu ne l'as pas."
)

def retrieve(query: str, k=5):
    q_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    passages = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        passages
