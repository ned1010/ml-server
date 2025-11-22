from uuid import uuid4
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import os
# Create an embedding
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI embeddding model


# create a embedding from a pdf
# takes the pdf file as input and returns the chunks
# clean the document

# Compile regex patterns once for better performance
EXCESSIVE_NEWLINES = re.compile(r'\n{3,}')
BULLET_POINTS = re.compile(r'([•\-\*])\s*([^\n]+)\n(?!\s*[•\-\*])')
SENTENCE_JOINS = re.compile(r'([^\.\!\?\n])\n(?=[a-z])')
ROMAN_NUMERALS = re.compile(r'\n([ivxlcdm]+)\.\s*', re.IGNORECASE)
WHITESPACE = re.compile(r' +')
PARAGRAPH_BREAKS = re.compile(r'\n\n+')


def clean_pdf_text(text: str) -> str:
    """OPTIMIZED: Clean and normalize PDF text with pre-compiled regex"""
    # Use pre-compiled regex patterns for better performance
    text = EXCESSIVE_NEWLINES.sub('\n\n', text)
    text = BULLET_POINTS.sub(r'\1 \2 ', text)
    text = SENTENCE_JOINS.sub(r'\1 ', text)
    text = ROMAN_NUMERALS.sub(r' \1. ', text)
    text = WHITESPACE.sub(' ', text)
    text = PARAGRAPH_BREAKS.sub('\n\n', text)
    return text.strip()


def clean_pdf_text_batch(texts: List[str]) -> List[str]:
    """Clean multiple texts in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(clean_pdf_text, texts))


@lru_cache(maxsize=128)
def get_document_name_from_source(source_path: str) -> str:
    """Extract document name from source path (cached)"""
    return source_path.split('/')[-1].split('.')[0]


def get_document_name(pages):
    """Get document title with caching"""
    if not pages or not pages[0].metadata:
        return "unknown_document"

    title = pages[0].metadata.get('title')
    if title:
        return title

    source = pages[0].metadata.get('source', '')
    return get_document_name_from_source(source)


def add_meta_data(chunks, document_id: str, document_name: str):
    """OPTIMIZED: Add metadata with batch processing"""
    total_chunks = len(chunks)

    # Pre-generate UUIDs in batch for better performance
    chunk_ids = [str(uuid4()) for _ in range(total_chunks)]

    # Use list comprehension for faster processing
    for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
        chunk.metadata.update({
            'document_id': document_id,
            'document_name': document_name,
            'chunk_id': chunk_id,
            'chunk_size': len(chunk.page_content),
            'chunk_overlap': 200,
            'total_chunk': total_chunks
        })
    return chunks


def add_meta_data_parallel(chunks, document_id: str, document_name: str):
    """Add metadata using parallel processing for large chunk sets"""
    if len(chunks) < 100:  # Use regular method for small sets
        return add_meta_data(chunks, document_id, document_name)

    def add_metadata_batch(chunk_batch, start_idx):
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunk_batch):
            chunk.metadata.update({
                'document_id': document_id,
                'document_name': document_name,
                'chunk_id': str(uuid4()),
                'chunk_size': len(chunk.page_content),
                'chunk_overlap': 200,
                'total_chunk': total_chunks
            })
        return chunk_batch

    # Split chunks into batches for parallel processing
    batch_size = max(1, len(chunks) // 4)  # 4 batches
    batches = [chunks[i:i + batch_size]
               for i in range(0, len(chunks), batch_size)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(add_metadata_batch, batch, i * batch_size)
                   for i, batch in enumerate(batches)]

        # Flatten results
        result_chunks = []
        for future in futures:
            result_chunks.extend(future.result())

    return result_chunks


def create_chunks_from_pdf_optimized(pdfPath: str, document_id: str):
    """OPTIMIZED: Create chunks with parallel processing and performance improvements"""
    start_time = time.time()

    # Load PDF
    print(f"📄 Loading PDF from: {pdfPath}")
    loader = PyPDFLoader(pdfPath)
    pages = loader.load()

    if not pages:
        print("❌ No pages loaded from PDF")
        return []

    load_time = time.time() - start_time
    print(f"⏱️  PDF loaded in {load_time:.2f}s ({len(pages)} pages)")

    # Clean text in parallel for better performance
    clean_start = time.time()
    if len(pages) > 4:  # Use parallel processing for larger documents
        page_texts = [page.page_content for page in pages]
        cleaned_texts = clean_pdf_text_batch(page_texts)
        for page, cleaned_text in zip(pages, cleaned_texts):
            page.page_content = cleaned_text
    else:
        # Sequential for small documents (less overhead)
        for page in pages:
            page.page_content = clean_pdf_text(page.page_content)

    clean_time = time.time() - clean_start
    print(f"🧹 Text cleaned in {clean_time:.2f}s")

    # Create text splitter with optimized settings
    chunk_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,  # Explicit length function
        is_separator_regex=False  # Faster for simple separators
    )

    # Split documents
    chunks = text_splitter.split_documents(pages)
    chunk_time = time.time() - chunk_start
    print(f"✂️  Text split in {chunk_time:.2f}s ({len(chunks)} chunks)")

    # Add metadata (use parallel for large chunk sets)
    meta_start = time.time()
    document_title = get_document_name(pages)

    if len(chunks) > 100:
        chunks = add_meta_data_parallel(chunks, document_id, document_title)
        print(f"📊 Metadata added in parallel")
    else:
        chunks = add_meta_data(chunks, document_id, document_title)

    meta_time = time.time() - meta_start
    total_time = time.time() - start_time

    # Performance summary
    print(f"\n✅ CHUNKING COMPLETE:")
    print(f"   📊 Total chunks: {len(chunks)}")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    print(f"   🚀 Chunks/second: {len(chunks)/total_time:.1f}")
    print(f"   📄 Pages/second: {len(pages)/total_time:.1f}")

    if chunks:
        print(f"\n📝 First chunk preview: {chunks[0].page_content[:200]}...")

    return chunks


def create_chunks_from_pdf(pdfPath: str, document_id: str):
    """LEGACY: Original method (kept for compatibility)"""
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


if __name__ == "__main__":
    user_id = "user_123"
    document_id = "9e1510be25d3ea9fd2b95b6fb81f5d1e4f1187cfdeb7bc3b62b97911deb8afd1"
    aws_pdf_url = "https://intreli.s3.ap-southeast-1.amazonaws.com/uploads/user_34lUBTwSTiKNSxq1KUS0lLZEKL8/1763621029120-NSW_Traineeship_Policies_and_Procedures_2025.pdf"

    print("Starting PDF processing...")
    print(f"Processing PDF from: {aws_pdf_url}")

    try:
        chunks = create_chunks_from_pdf_optimized(aws_pdf_url, document_id)
        print(chunks)

        # Show sample of chunks
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.page_content[:150]}...")
            print(f"Metadata: {chunk.metadata}")

    except Exception as e:
        print(f"Error processing PDF: {e}")
