import os, re
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
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

# Embeddings + OpenAI
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Base Chroma (remplace FAISS)
chroma_client = chromadb.Client()
collection = chromadb.Client().create_collection("technopure")

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","noscript","header","footer","form","nav","aside"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return
