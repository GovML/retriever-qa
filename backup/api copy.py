# api.py
import requests
import streamlit as st

st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("📄 PDF Q&A Search")

question = st.text_input("Ask a question about the documents:")

if st.button("Search") and question.strip():
    with st.spinner("Searching..."):
        try:
            # Step 1: Query the API for top-k documents
            query_response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"question": question, "top_k": 2}
            )

            if query_response.status_code != 200:
                st.error(f"❌ Query request failed with status code {query_response.status_code}")
                st.stop()

            query_results = query_response.json()
            document_names = [doc["name"] for doc in query_results["results"]]

            # Step 2: Synthesize the answer using selected documents
            synth_response = requests.post(
                "http://127.0.0.1:8000/synthesize",
                json={"question": question, "document_names": document_names}
            )

            if synth_response.status_code != 200:
                st.error(f"❌ Synthesis request failed with status code {synth_response.status_code}")
                st.stop()

            synth_result = synth_response.json()

            # Step 3: Display the synthesized answer
            st.subheader("🧠 Synthesized Answer")
            st.write(synth_result["answer"])

            # Step 4: Show sources
            st.subheader(f"📚 Top Results for: *{query_results['question']}*")
            for i, doc in enumerate(query_results["results"]):
                with st.expander(f"Result #{i + 1} — {doc['name']} (Score: {doc['score']:.4f})"):
                    st.write(doc["content"])

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
