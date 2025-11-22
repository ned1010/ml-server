# create vector store and upload to the pinecone
import os
# from embeddings import create_chunks_from_pdf

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores_sparse import PineconeSparseVectorStore
from langchain_pinecone.embeddings import PineconeSparseEmbeddings


from concurrent.futures import ThreadPoolExecutor
import asyncio
# use hash for index of the user name
import hashlib


# testing purposes
# from embeddings import create_chunks_from_pdf

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
    return OpenAIEmbeddings(api_key=api_key, model='text-embedding-ada-002')


def get_sparse_embedding():
    """Initialize Pinecone sparse embeddings with error handling"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    return PineconeSparseEmbeddings(api_key=api_key, model="pinecone-sparse-english-v0")


# create embedding and store to pinecone database
# create dense and sparse index


def create_vector_store(chunks, user_id):
    """
    Create both dense and sparse vector stores and add chunks to Pinecone
    Returns: (dense_vector_store, sparse_vector_store)
    """
    dense_index_name = 'intreli-dense'
    sparse_index_name = 'intreli-sparse'

    try:
        # Initialize clients
        pc = get_pinecone_client()
        dense_embedding = get_openai_embeddings()
        sparse_embedding = get_sparse_embedding()

        print(f"Processing {len(chunks)} chunks for user: {user_id}")

        # existing names
        existing_indexes = pc.list_indexes().names()

        # === PARALLEL INDEX CREATION ===
        def create_dense_index():
            if dense_index_name not in existing_indexes:
                print(f"Creating dense index: {dense_index_name}")
                pc.create_index(
                    name=dense_index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                print(f"Dense index {dense_index_name} created successfully")
            else:
                print(f"Dense index {dense_index_name} already exists")

        def create_sparse_index():
            if sparse_index_name not in existing_indexes:
                print(f"Creating sparse index: {sparse_index_name}")
                pc.create_index(
                    name=sparse_index_name,
                    vector_type='sparse',
                    metric='dotproduct',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                print(f"Sparse index {sparse_index_name} created successfully")
            else:
                print(f"Sparse index {sparse_index_name} already exists")

        # Run index creation in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(create_dense_index)
            sparse_future = executor.submit(create_sparse_index)

            # Wait for both to complete
            dense_future.result()
            sparse_future.result()

        # === PARALLEL VECTOR STORE CREATION ===
        def create_dense_store():
            dense_vector_store = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=dense_embedding,
                index_name=dense_index_name,
                namespace=user_id
            )
            print(f"Added {len(chunks)} chunks to dense index")
            return dense_vector_store

        def create_sparse_store():
            sparse_vector_store = PineconeSparseVectorStore.from_documents(
                documents=chunks,
                embedding=sparse_embedding,
                index_name=sparse_index_name,
                namespace=user_id
            )
            print(f"Added {len(chunks)} chunks to sparse index")
            return sparse_vector_store

        # Run vector store creation in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_store_future = executor.submit(create_dense_store)
            sparse_store_future = executor.submit(create_sparse_store)

            # Get results
            dense_vector_store = dense_store_future.result()
            sparse_vector_store = sparse_store_future.result()

        print("Both vector stores created successfully!")
        return dense_vector_store, sparse_vector_store

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None


# if __name__ == "__main__":
#     # Test the vector store functions
#     user_id = "user_123"
#     document_id = "9e1510be25d3ea9fd2b95b6fb81f5d1e4f1187cfdeb7bc3b62b97911deb8afd1"
#     aws_pdf_url = "https://intreli.s3.ap-southeast-1.amazonaws.com/uploads/user_34lUBTwSTiKNSxq1KUS0lLZEKL8/1763621029120-NSW_Traineeship_Policies_and_Procedures_2025.pdf"

#     print("Testing vector store operations...")

#     # Create chunks from PDF
#     print("\n1. Creating chunks from PDF...")
#     chunks = create_chunks_from_pdf(aws_pdf_url, document_id)
#     print(f"Created {len(chunks)} chunks")

#     # Show first chunk as sample
#     if chunks:
#         print(f"\nSample chunk:")
#         print(f"Content: {chunks[0].page_content[:200]}...")
#         print(f"Metadata: {chunks[0].metadata}")

#     # Create vector stores using OPTIMIZED method
#     print("\n2. Creating vector stores (OPTIMIZED)...")
#     dense_store, sparse_store = create_vector_store(
#         chunks, user_id)

#     if chunks:
#         test_query = "What is this document about?"
#         # Create vector store with chunks
#         if dense_store:
#             # Test search
#             results = dense_store.similarity_search_with_score(test_query, k=5)

#             print(results)
#     #         results = dense_store.similarity_search_with_score(test_query, k=5)

#     #         print(results)

#         if sparse_store:
#             # Test search
#             results = sparse_store.similarity_search_with_score(
#                 test_query, k=5)

#             print(results)
