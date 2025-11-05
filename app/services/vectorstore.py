#create vector store and upload to the pinecone
import os
# from embeddings import create_chunks_from_pdf

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

#use hash for index of the user name
import hashlib

load_dotenv()

def get_pinecone_client():
    """Initialize Pinecone client with error handling"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    return Pinecone(api_key=api_key)

def get_openai_embeddings():
    """Initialize OpenAI embeddings with error handling"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAIEmbeddings(api_key=api_key)



#create embedding and store to pinecone database
def create_vector_store(chunks, index_name):
    """
    Create a vector store and add chunks to Pinecone
    """
    try:
        # Initialize clients
        pc = get_pinecone_client()
        embed = get_openai_embeddings()
        
        # Check if index exists, if not create it
        existing_indexes = [hashlib.md5(index.name.encode()).hexdigest() for index in pc.list_indexes()]
        
        if hashlib.md5(index_name.encode()).hexdigest() not in existing_indexes:
            print(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name, 
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine", 
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Index {index_name} created successfully")

            #create a vector store
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embed,
                index_name=index_name
            )

        else:
            #add to existing index
            print(f"Index {index_name} already exists")
           # Connect to existing index
           
           #check if the chunks exists in the index
           #hashing method to check the duplication of the data
           #TODO: Vector Duplication 
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embed
            )
            
            # Add new documents
            vectorstore.add_documents(chunks)
            
            print(f"Successfully added {len(chunks)} more chunks to existing index: {index_name}")
        return vectorstore
        
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


# if __name__ == "__main__":
#     # Test the vector store functions
#     aws_pdf_url = 'https://intreli.s3.ap-southeast-1.amazonaws.com/uploads/user_34lUBTwSTiKNSxq1KUS0lLZEKL8/1761986074058-1761985252258-file-example_PDF_500_kB.pdf'

    
#     print("Testing vector store operations...")
    
#     # Create chunks from PDF
#     chunks = create_chunks_from_pdf(aws_pdf_url)
#     vectorstore = create_vector_store(chunks, 'test123', embed)
    
#     if chunks:
#         # Create vector store with chunks
#         if vectorstore:
#             # Test search
#             test_query = "What is this document about?"
#             results = vectorstore.similarity_search(test_query, k=5)
            
#             for i, result in enumerate(results[:2]):
#                 print(f"\nResult {i+1}: {result.page_content[:200]}...")