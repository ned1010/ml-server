
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import hashlib
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeSparseEmbeddings, PineconeSparseVectorStore
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from itertools import chain
import os
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
router = APIRouter()


# eventually add document id
class ChatRequest(BaseModel):
    userId: str
    question: str
    # documentId: str


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
dense_embedding_model = OpenAIEmbeddings(api_key=os.getenv(
    "OPENAI_API_KEY"), model="text-embedding-ada-002")

sparse_embeddings = PineconeSparseEmbeddings(api_key=os.getenv(
    "PINECONE_API_KEY"), model="pinecone-sparse-english-v0")


def merge_sparse_dense_chunks(chunk1, chunk2):
    # ids and merged
    seen_ids = set()
    merged_chunks = []

    for (docs, scores) in chain(chunk1, chunk2):
        # print(docs)
        # print(scores
        if docs.metadata.get('chunk_id') not in seen_ids:
            seen_ids.add(docs.metadata.get('chunk_id'))

            document = {
                'page_content': docs.page_content,
                'metadata': docs.metadata,
                'score': scores

            }

            merged_chunks.append(document)
    return merged_chunks


def threshold_reranked_chunks(
    documents,
    threshold: float = 0.65,
    min_fallback_score: float = 0.5,
    fallback: bool = True
):
    """
    Selects reranked chunks above the threshold.
    Falls back to highest scoring document if configured.

    documents: output from reranker (documents.data)
               Expected: [{'text': ..., 'score': ...}, ...]
    """

    # Defensive: handle None or empty
    if not documents or not hasattr(documents, "data") or not documents.data:
        return []

    docs = documents.data

    # 1. Filter by threshold
    relevant = [doc for doc in docs if doc.get("score", 0) >= threshold]

    if relevant:
        return relevant                      # Primary happy-path

    # 2. Optional fallback
    if fallback:
        top_doc = max(docs, key=lambda d: d.get("score", 0))

        if top_doc.get("score", 0) >= min_fallback_score:
            return [top_doc]

    # 3. Final fallback: return all (or empty — your choice)
    return docs


def generate_sources(relevant_chunks, include_chunk_text=True):
    sources = []
    for i, chunk in enumerate(relevant_chunks, 1):
        source = {
            'label': f"[{i}]",
            'pdf_name': chunk.document.metadata.get('document_name', 'Unknown'),
            'page': chunk.document.metadata.get('page', 'N/A'),
            'chunk_id': chunk.document.metadata.get('chunk_id', ''),
            'score': getattr(chunk, 'score', 0)
        }
        if include_chunk_text:
            source['chunk_text'] = chunk.document.page_content
        sources.append(source)
    return sources


def prepare_context_with_citations(relevant_chunks):
    context_parts = []
    citations = []

    for i, row in enumerate(relevant_chunks, 1):
        # Create citation metadata
        citation = {
            'label': f"[{i}]",
            'pdf_name': row.document.metadata.get('document_name', 'Unknown'),
            'page': row.document.metadata.get('page', 'N/A'),
            'chunk_id': row.document.metadata.get('chunk_id', ''),
            'score': getattr(row, 'score', 0)
        }
        citations.append(citation)

        # Format context with citation marker
        context_part = f"[{i}] {row.document.metadata.get('document_name', 'Document')} — Page {row.document.metadata.get('page', 'N/A')}\n{row.document.page_content}"
        context_parts.append(context_part)

    context = '\n\n'.join(context_parts)
    return context, citations

# TODO: Change the citation to include from Intreli


def create_citation_aware_prompt(context, citations, query):
    """Create prompt that encourages proper citation usage"""

    citation_instructions = """
You are a precise document assistant. Answer questions using ONLY the provided context.

CORE RULES:
1. Use only information explicitly stated in the context
2. Cite every claim immediately: [1], [2], [3]
3. If information is not in context, say: "This information is not found in the provided documents"
5. Answer only what is asked - do not summarize the entire document

CRITICAL: Avoid starting with generic phrases like "The document is about..." Get straight to the question.

CITATION FORMAT:
- Place citations immediately after claims: "The system uses OAuth [1]."
- Combine sources for the same claim if needed: [1,2]
- Maximum 3 citations per sentence
- Direct quotes require citations: "exact text" [1]


ANSWER STRUCTURE:
- Each sentence should make ONE specific point
- Cite 1-2 sources per sentence (max 3 if absolutely necessary)

PROHIBITED:
- Adding info not in context
- Vague qualifiers unless in the source
- Document overviews unless asked
- Over-citing (>3 sources per sentence)

If you cannot answer with confidence using only the context, explicitly state what information is missing.

Available Sources:
"""

    # Add source list for reference
    source_list = "\n".join([
        f"{cite['label']} {cite['pdf_name']} — Page {cite['page']}"
        for cite in citations
    ])

    full_prompt = f"""
{citation_instructions}
{source_list}

Context with Sources:
{context}

Question: {query}

Please provide a comprehensive answer with proper inline citations.
"""

    return full_prompt


@router.post("/streamchat")
async def chat_stream(request: ChatRequest):
    print("User ID received", request.userId)

    async def generate():
        try:
            # Create a hash of the user_id
            # hash_value = hashlib.md5(request.userId.encode()).hexdigest()
            # index_name = f"{hash_value}-docs"

            # index names
            dense_index_name = 'intreli-dense'
            sparse_index_name = 'intreli-sparse'

            # Check if the index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            if dense_index_name not in existing_indexes:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Dense Index not found'})}\n\n"
                return

            if sparse_index_name not in existing_indexes:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Sparse Index not found'})}\n\n"
                return

            # Get the pinecone index
            dense_index = pc.Index(dense_index_name)
            sparse_index = pc.Index(sparse_index_name)

            # Connect to the dense and sparse index

            # Create LangChain vectorstore from the pinecone index
            dense_vectorstore = PineconeVectorStore(
                index=dense_index, embedding=dense_embedding_model)
            sparse_vectorstore = PineconeSparseVectorStore(
                index=sparse_index, embedding=sparse_embeddings)
            # print(dense_vectorstore)
            # print(sparse_vectorstore)

            # Question
            question = request.question

            # Use text-based search methods (not vector-based)
            # Send status update
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"
            # dense_result = dense_vectorstore.similarity_search_with_score(
            #     question, k=20, namespace=request.userId, filter={"document_id": request.documentId})

            dense_result = dense_vectorstore.similarity_search_with_score(
                question, k=20, namespace=request.userId)

            # Send status update
            # yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"
            # sparse_result = sparse_vectorstore.similarity_search_with_score(
            #     question, k=20, namespace=request.userId, filter={"document_id": request.documentId})

            sparse_result = sparse_vectorstore.similarity_search_with_score(
                question, k=20, namespace=request.userId)

            merged = merge_sparse_dense_chunks(dense_result, sparse_result)
            # print(merged)

            # rerank the results
            reranked_chunks = pc.inference.rerank(model='bge-reranker-v2-m3', query=question,
                                                  documents=merged, rank_fields=['page_content'], top_n=5, return_documents=True)

            # filter the results
            relevant_chunks = threshold_reranked_chunks(reranked_chunks, 0.65)
            # print(relevant_chunks)

            sources = generate_sources(relevant_chunks)
            # Send sources immediately
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'count': len(relevant_chunks)})}\n\n"

            context, citation = prepare_context_with_citations(relevant_chunks)

            full_prompt = create_citation_aware_prompt(
                context, citation, query=question)

            # Send status update
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"

            # Stream the response from the model
            full_response = ""
            async for chunk in model.astream(full_prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            # Send completion signal with sources included
            completion_data = {
                'type': 'done',
                'full_response': full_response,
                'sources': sources,
                'citations': citation,
                'total_sources': len(relevant_chunks)
            }
            yield f"data: {json.dumps(completion_data)}\n\n"

        except HTTPException as e:
            yield f"data: {json.dumps({'type': 'error', 'error': e.detail})}\n\n"
        except Exception as e:
            print(f"Error in chat stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'An error occurred while processing your request'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
