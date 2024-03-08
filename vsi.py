from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents and build index
documents = SimpleDirectoryReader(
    "data/pdf"
).load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True)