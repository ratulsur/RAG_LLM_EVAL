class HybridRetriever:
    def __init__(self, documents, alpha = 0.5):
        """

        alpha: BM25 weights
        (1-alpha): dense retrieval weights        
        """

        self.documents = documents
        self.alpha = alpha

        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)


        self.embedding_model = (SentenceTransformer("all-miniLM-L6-v2"))

        self.doc_embeddings = (self.embedding_model.encode(documents, normalize_embeddings= True, convert_to_numpy=True))

        self.cross_encoder = (CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2"))

    def _normalize(self, scores):
        scores = np.array(scores)

        if scores.max()==scores.min():
            return np.ones_like(score)
        return (scores - scores.min())/ (scores.max() - scores.min())

def hybrid_search(self, query, top_k = 50):
    query_tokens = (query.lower().split())
    bm25_scores = (self.bm25.get_scores(query_tokens))
    query_embedding = (self.embedding_model.encode(query,normalize_embeddings = True, convert_to_numpy = True))

    dense_scores = np.dot(self.doc_embeddings, query_embedding)

    bm25_scores = (self._normalize(bm25_scores))
    dense_scores = (self._normalize(dense_scores))

    hybrid_scores = (self.alpha * bm25_scores + (1-self.alpha) * dense_scores)

    ranked_indices = np.argsort(hybrid_scores)[::1]

    results = []

    for idx in ranked_indices[:top_k]:
        results.append({
            "doc_id": idx,
            "document": self.documents[idx],
            "embedding": self.embedding[idx],
            "hybrid_score": float(bm25_scores[idx]),
            "dense_scores": float(dense_scores[idx])
        })

        return results
    def mmr(self, query, retrieved_docs, top_k = 15, lambda_parameter = 0.7):
        query_embedding = (self.embedding_model.encode(query, normalize_embedding = True, convert_to_numpy = True))
        doc_embedding = np.array([d["embedding"] for d in retrieved_docs])
        query_similaries = np.dot(doc_embedding, query_embedding)

        selected = []
        remaining = list(range(len(retrieved_docs)))

        first_idx = np.argmax(query_similaries)
        selected.append(first_idx)
        remaining.remove(first_idx)

        #---------------------------
        # GREEDY MMR
        #---------------------------

        while (len(selected) < top_k and remaining):
            best_score = -np.inf
            best_doc = None

            for idx in remaining:
                relevance = (query_similaries[idx])
                redundancy = max(np.dot(doc_embedding[idx], doc_embedding[s]) for s in selected)

                mmr_score = (lambda_parameter * relevance - (1 - lambda_parameter)* redundancy)

                if (mmr_score > best_score):
                    best_score = mmr_score
                    best_doc = idx

                    selected.append(best_doc)
                    remaining.remove(best_doc)

                return [retrieved_docs[i] for i in selected]
            
        #--------------------------
        #CROSS ENCODER
        #--------------------------

        def rerank(self, query, docs, top_k = 5):
            pairs = [(query, d["documents"]) for d in docs]

            scores = (self.cross_encoders.predict(pairs))

            for doc, scores in zip(doc, scores):
                doc["cross_encoder_score"] = float(score)

                docs.sort(key = lambda x: x["cross_encoder_score"], reverse = True)
                return docs [:top_k]
            
            











