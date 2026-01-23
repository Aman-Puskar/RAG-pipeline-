import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# cleaning of the text extracted from the source
def clean_text(text):
    if not text:
        return text
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text.strip()


# loading all the pdfs from the folder
def load_all_pdf(folder):
    documents = []
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    print(f"\n Total pdfs are {len(pdf_files)}")

    for file in pdf_files:
        pdf_path = os.path.join(folder, file)
        print(f'\nloading text from {file}')
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # improving metadata
        for i, p in enumerate(pages):
            p.metadata["source"] = file
            p.metadata["page"] = i + 1
            p.page_content = clean_text(p.page_content)
    
        documents.extend(pages)
        
    print(f"Total pages extracted from the source : {len(documents)}")
    return documents    


# creation of the chunks
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    for idx, c in enumerate(chunks, start=1):
        c.metadata["chunk_id"] = idx
    return chunks


# creation of embeddings and storing 

pc = Pinecone(api_key="pcsk_2tp4k5_FVDovQsf9PbviQ6FJSPi7Enxxog2yzxBJC7cWNFFmQnm6udLMAYNrzAgKV1KVLY")
index_name = "rag-index"

# Create index if not exists
if index_name not in [i['name'] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        )

    )

index = pc.Index(index_name)

def store_in_pinecone(chunks):
    vectors = []

    for chunk in chunks:
        embedding = embedding_model.embed_documents(chunk.page_content)

        vector_data = {
            "id": f"chunk_{chunk.metadata['chunk_id']}",
            "values": embedding,
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source"),
                "page": chunk.metadata.get("page"),
                "chunk_id": chunk.metadata.get("chunk_id")
            }
        }

        vectors.append(vector_data)

    print(f"\nUploading {len(vectors)} vectors to Pinecone...")

    for i in range(0, len(vectors), 100):
        batch = vectors[i : i + 100]
        index.upsert(vectors=batch)

    print("All vectors uploaded successfully!")

documents = load_all_pdf("sampleFolder")
chunks = create_chunks(documents)
store_in_pinecone(chunks)
