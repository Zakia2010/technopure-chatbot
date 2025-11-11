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

app = FastAPI(title="Technopure Site Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.technopure.ma", "https://technopure.ma", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- OpenAI + Chroma ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("technopure")

# ---------- Embeddings ----------
def embed_texts(texts):
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [r.embedding for r in res.data]

# ---------- Extraction site ----------
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

# ---------- Indexation manuelle ----------
def build_index():
    pages = crawl()
    docs, metas = [], []
    for url, text in pages:
        for part in chunk(text):
            docs.append(part)
            metas.append({"url": url})
    if not docs:
        return {"status": "Aucune page trouvée"}
    embeddings = embed_texts(docs)
    ids = [str(i) for i in range(len(docs))]
    collection.add(documents=docs, metadatas=metas, embeddings=embeddings, ids=ids)
    print(f"✅ {len(docs)} passages indexés depuis technopure.ma")
    return {"status": f"{len(docs)} passages indexés"}

@app.api_route("/reindex", methods=["GET", "POST"])
def reindex():
    """Reconstruit manuellement l'index à partir du site Technopure"""
    return build_index()

# ---------- Recherche et génération ----------
SYSTEM_PROMPT = (
    "Tu es l'assistant Technopure. Réponds UNIQUEMENT avec les informations "
    "du contexte (extraits de technopure.ma). Si tu ne sais pas, dis-le simplement."
)

def retrieve(query: str, k=5):
    q_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    passages = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        passages.append({"text": text, "url": meta["url"], "score": 1 - dist})
    return passages

def generate_answer(query, passages):
    if not passages:
        return {
            "answer": "Désolé, je ne trouve pas cette information sur technopure.ma.",
            "sources": [],
        }
    context = "\n\n".join([f"[{p['url']}]\n{p['text']}" for p in passages])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContexte:\n{context}\n\nRéponds en français et cite les URLs."},
    ]
    rsp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    return {"answer": rsp.choices[0].message.content, "sources": list({p["url"] for p in passages})}

@app.post("/chat")
def chat(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        return {"answer": "Posez votre question sur le contenu de technopure.ma.", "sources": []}
    passages = retrieve(q)
    return generate_answer(q, passages)

@app.get("/")
def home():
    return {"status": "✅ Technopure Chatbot API is running", "endpoints": ["/reindex", "/chat"]}
