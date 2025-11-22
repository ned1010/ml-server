# Create an embedding
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from uuid import uuid4
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI embeddding model


# create a embedding from a pdf
# takes the pdf file as input and returns the chunks
# clean the document

def clean_pdf_text(text):
    """Clean and normalize PDF text"""

    # 1. Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 2. Join bullet points with their content
    # Handles: • item\n content → • item content
    text = re.sub(r'([•\-\*])\s*([^\n]+)\n(?!\s*[•\-\*])', r'\1 \2 ', text)

    # 3. Join lines that are part of same sentence
    # If line doesn't end with punctuation, join with next
    text = re.sub(r'([^\.\!\?\n])\n(?=[a-z])', r'\1 ', text)

    # 4. Fix Roman numerals (i., ii., iii., iv., etc.)
    text = re.sub(r'\n([ivxlcdm]+)\.\s*', r' \1. ', text, flags=re.IGNORECASE)

    # 5. Normalize whitespace
    text = re.sub(r' +', ' ', text)

    # 6. Keep paragraph breaks
    text = re.sub(r'\n\n+', '\n\n', text)

    return text.strip()


def get_document_name(pages):
    # TODO: create a better implementation to get title of a PDF file

    if not pages[0].metadata.get('title'):
        # Getting the document title from the source path
        doc_title = pages[0].metadata['source'].split('/')[-1].split('.')[0]
    else:
        doc_title = pages[0].metadata['title']
    return doc_title


def add_meta_data(chunks, document_id, document_name):
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            'document_id': document_id,
            'document_name': document_name,
            'chunk_id': str(uuid4()),
            'chunk_size': len(chunk.page_content),
            'chunk_overlap': 200,
            'total_chunk': len(chunks)
        })
    return chunks


def create_chunks_from_pdf(pdfPath, document_id):
    loader = PyPDFLoader(pdfPath)
    pages = loader.load()

    for page in pages:
        page.page_content = clean_pdf_text(page.page_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=200)

    # create chunks
    chunks = text_splitter.split_documents(pages)

    # add meta data
    document_title = get_document_name(pages)

    chunks = add_meta_data(chunks, document_id, document_title)

    # create embeddings
    print(f"Created {len(chunks)} chunks from PDF")
    print(
        f"First chunk preview: {chunks[0].page_content[:200]}..." if chunks else "No chunks created\n")

    # send teh chunks to vectorstore
    return chunks


# if __name__ == "__main__":
#     user_id = "user_123"
#     document_id = "9e1510be25d3ea9fd2b95b6fb81f5d1e4f1187cfdeb7bc3b62b97911deb8afd1"
#     aws_pdf_url = "https://intreli.s3.ap-southeast-1.amazonaws.com/uploads/user_34lUBTwSTiKNSxq1KUS0lLZEKL8/1763621029120-NSW_Traineeship_Policies_and_Procedures_2025.pdf"

#     print("Starting PDF processing...")
#     print(f"Processing PDF from: {aws_pdf_url}")

#     try:
#         chunks = create_chunks_from_pdf(aws_pdf_url, document_id)

#         # Show sample of chunks
#         for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
#             print(f"\nChunk {i+1}:")
#             print(f"Content: {chunk.page_content[:150]}...")
#             print(f"Metadata: {chunk.metadata}")

#     except Exception as e:
#         print(f"Error processing PDF: {e}")
