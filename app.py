import os, re, json, requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from fastapi import FastAPI, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# ---------- CONFIGURATION ----------
BASE = "https://www.technopure.ma/"
ALLOWED_NETLOC = urlparse(BASE).netloc.replace("www.", "")

app = FastAPI(title="Technopure Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://technopure.ma",
        "https://www.technopure.ma",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- INITIALISATION DES CLIENTS ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client(Settings(persist_directory="/tmp/chroma_data"))

try:
    collection = chroma_client.get_collection("technopure")
except:
    collection = chroma_client.create_collection("technopure")

# ---------- EMBEDDINGS ----------
def embed_texts(texts):
    """Crée les embeddings via OpenAI."""
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [r.embedding for r in res.data]

# ---------- EXTRACTION DU CONTENU DU SITE ----------
def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","header","footer","form","nav","aside"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

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

def crawl(start=BASE, limit=150):
    """Explore le site Technopure pour en extraire le contenu."""
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
            pages.append({"url": url, "text": text})
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                u = urljoin(url, a["href"])
                p = urlparse(u)
                host = p.netloc.replace("www.", "")
                if host == ALLOWED_NETLOC and p.scheme in ("http", "https"):
                    u = u.split("#")[0]
                    if not u.endswith((".pdf", ".zip", ".jpg", ".png", ".jpeg", ".gif", ".webp")):
                        queue.append(u)
        except Exception:
            continue
    return pages

# ---------- INDEXATION MANUELLE (LOCAL OU DIRECTE) ----------
def build_index(pages=None):
    """Construit ou met à jour l’index Chroma."""
    if not pages:
        pages = crawl()

    docs, metas = [], []
    for p in pages:
        for part in chunk(p["text"]):
            docs.append(part)
            metas.append({"url": p["url"]})

    if not docs:
        return {"status": "Aucune page trouvée"}

    embeddings = embed_texts(docs)
    ids = [str(i) for i in range(len(docs))]
    collection.add(documents=docs, metadatas=metas, embeddings=embeddings, ids=ids)
    return {"status": f"✅ {len(docs)} passages indexés avec succès."}

@app.api_route("/reindex", methods=["GET", "POST"])
def reindex():
    """Reconstruit manuellement l'index depuis technopure.ma (lent)."""
    return build_index()

@app.post("/upload_index")
async def upload_index(file: UploadFile = File(...)):
    """Permet d’importer un fichier JSON d’indexation créé localement."""
    try:
        data = json.loads(await file.read())
        return build_index(data)
    except Exception as e:
        return {"error": str(e)}

# ---------- CHATBOT ----------
SYSTEM_PROMPT = (
    "Tu es l'assistant Technopure. Réponds UNIQUEMENT avec les informations "
    "trouvées sur technopure.ma. Si tu ne sais pas, dis-le simplement."
)

def retrieve(query: str, k=5):
    if collection.count() == 0:
        return [{"text": "Aucune donnée indexée pour le moment.", "url": "https://technopure.ma", "score": 1.0}]
    q_emb = embed_texts([query])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    passages = []
    for text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        passages.append({"text": text, "url": meta["url"], "score": 1 - dist})
    return passages

def generate_answer(query, passages):
    if not passages:
        return {"answer": "Désolé, je ne trouve pas cette information sur technopure.ma.", "sources": []}

    context = "\n\n".join([f"[{p['url']}]\n{p['text']}" for p in passages])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContexte:\n{context}\n\nRéponds en français et cite les URLs."},
    ]

    try:
        rsp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
        return {"answer": rsp.choices[0].message.content, "sources": list({p["url"] for p in passages})}
    except Exception as e:
        return {"answer": f"⚠️ Erreur lors de la génération de réponse : {e}", "sources": []}

@app.post("/chat")
def chat(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        return {"answer": "Posez votre question sur le contenu de technopure.ma.", "sources": []}
    passages = retrieve(q)
    return generate_answer(q, passages)

# ---------- PAGE D’ACCUEIL ----------
@app.get("/")
def home():
    return {
        "status": "✅ Technopure Chatbot API is running",
        "endpoints": ["/reindex", "/upload_index", "/chat"],
        "message": "Bienvenue sur l'API Chatbot de Technopure 🚀"
    }
