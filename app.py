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

# ❌ Supprimer cette ligne (elle crée la surcharge mémoire) :
# build_index()

# ✅ Ajoute cette route manuelle :
@app.get("/reindex")
def reindex():
    """Reconstruit l'index manuellement depuis technopure.ma"""
    return build_index()

@app.post("/chat")
def chat(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        return {
            "answer": "Posez votre question sur le contenu de technopure.ma.",
            "sources": [],
        }
    passages = retrieve(q)
    return generate_answer(q, passages)
