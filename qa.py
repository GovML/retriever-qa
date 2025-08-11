import json
import numpy as np
from typing import Optional, List

from haystack import Document
from haystack.core.pipeline import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from deepseek_client import DeepSeekClient




class PDFRetriever:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.documents = []
        self.doc_store = None
        self.pipeline = None

        self._load_json()
        self._initialize_docstore()
        self._build_pipeline()
        self.client = DeepSeekClient(model_name="deepseek-r1:7b", temperature=0.6)

    def _load_json(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def _initialize_docstore(self):
        print("Initializing InMemoryDocumentStore...")
        self.doc_store = InMemoryDocumentStore()

        all_docs = []
        first_item = list(self.data.items())[0]
        print("First document key and structure:", first_item[0], type(first_item[1]))

        for doc_name, doc_content in self.data.items():
            # doc_content is a dict of chunks: {"chunk_0": {...}, "chunk_1": {...}}
            for chunk_id, chunk_data in doc_content.items():
                if not isinstance(chunk_data, dict):
                    print(f"Skipping malformed chunk {chunk_id} in {doc_name}")
                    continue

                if "embedding" not in chunk_data or "text" not in chunk_data:
                    print(f"Warning: No embedding/text found for {doc_name} - {chunk_id}, skipping.")
                    continue

                embedding = np.array(chunk_data["embedding"], dtype=np.float32)
                text = chunk_data["text"]

                haystack_doc = Document(
                    content=text,
                    embedding=embedding,
                    meta={
                        "name": doc_name,
                        "chunk_id": chunk_id
                    }
                )
                all_docs.append(haystack_doc)

        print(f"Writing {len(all_docs)} documents to the store...")
        self.doc_store.write_documents(all_docs)

    def _build_pipeline(self):
        print("Building query pipeline...")
        embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        retriever = InMemoryEmbeddingRetriever(document_store=self.doc_store)

        self.pipeline = Pipeline()
        self.pipeline.add_component("text_embedder", embedder)
        self.pipeline.add_component("retriever", retriever)
        self.pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    def query(self, question: str, top_k: int = 5):
        print(f"\nQuerying for: {question}")
        result = self.pipeline.run({
            "text_embedder": {"text": question},
            "retriever": {"top_k": top_k}
        })

        documents = result["retriever"]["documents"]
        print([d for d in documents])
        return documents

    def synthesize(self, question: str, documents: List[Document]) -> str:
        """
        Synthesizes an answer from the provided list of Haystack `Document` objects using DeepSeekClient.
        """
        if not documents:
            return "No matching documents provided."

        context = ""
        for i, doc in enumerate(documents):
            doc_text = doc.content.strip().replace("\n", " ")
            context += f"[Source {i+1} - {doc.meta.get('name', 'unknown')}]: {doc_text}\n\n"

        prompt = f"""You are a citation-providing assistant answering questions using the content of the documents provided. 
The content you will be provided is long. First, find the section mentioning themes related to the question. Using those themes, then answer the question.
You must cite your source from within the content of the documents.

Here is the content of the document:

{context}

Question: {question}
Answer:"""

        response = self.client.generate(prompt)
        answer = response['text'] if isinstance(response, dict) and 'text' in response else str(response)
        return answer


# Optional API
def start_api(json_path: str, host: str = "127.0.0.1", port: int = 8000):
    retriever = PDFRetriever(json_path=json_path)
    app = FastAPI(title="PDF QA API")

    class QueryRequest(BaseModel):
        question: str
        top_k: Optional[int] = 5

    class SynthesizeRequest(BaseModel):
        question: str
        top_k: Optional[int] = 5

    @app.post("/query")
    def ask_question(query: QueryRequest):
        docs = retriever.query(query.question, top_k=query.top_k)
        return {
            "question": query.question,
            "results": [
                {
                    "name": doc.meta.get("name"),
                    "score": doc.score,
                    "content": doc.content,
                } for doc in docs
            ]
        }

    @app.post("/synthesize")
    def synthesize_answer(request: SynthesizeRequest):
        docs = retriever.query(request.question, top_k=request.top_k)
        answer = retriever.synthesize(request.question, documents=docs)
        return {
            "question": request.question,
            "answer": answer,
        }

    uvicorn.run(app, host=host, port=port)


# Usage: CLI or API
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="./tmp_extract_pdf/combined_output.json")
    parser.add_argument("--api", action="store_true", help="Run as API")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.api:
        start_api(json_path=args.json_path, host=args.host, port=args.port)
    else:
        print(f'Initializing Document Store from {args.json_path}')
        retriever = PDFRetriever(json_path=args.json_path)
        docs = retriever.query("What is the main purpose of the NAP for Albania?", top_k=3)
        print(retriever.synthesize("What is the main purpose of the NAP for Albania?", documents=docs))
