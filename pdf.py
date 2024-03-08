import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.readers import SimpleDirectoryReader


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("Creating index...")
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        print("Loading index...")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


canada_pdf = SimpleDirectoryReader("data/pdf").load_data()
canada_index = get_index(canada_pdf, "canada_index")
canada_engine = canada_index.as_query_engine()
