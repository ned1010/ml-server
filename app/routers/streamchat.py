
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import hashlib
import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
router = APIRouter()

class ChatRequest(BaseModel):
    userId: str
    question: str


model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
embed = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

@router.post("/streamchat")
async def chat_stream(request: ChatRequest):
    print("User ID received", request.userId)

    async def generate():
        try:
            # Create a hash of the user_id
            hash_value = hashlib.md5(request.userId.encode()).hexdigest()
            index_name = f"{hash_value}-docs"

            # Check if the index exists
            existing_indexes = [index.name for index in pc.list_indexes()]
            if index_name not in existing_indexes:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Index not found'})}\n\n"
                return

            # Get the pinecone index
            index = pc.Index(index_name)

            # Create LangChain vectorstore from the pinecone index
            vectorstore = PineconeVectorStore(index=index, embedding=embed)
            print(vectorstore)

            # Question
            question = request.question

            # Send status update
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching documents...'})}\n\n"

            # Retrieve documents
            retrieved_docs = vectorstore.similarity_search(question, k=5)

            # Extract sources
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A")
                }
                for doc in retrieved_docs
            ]

            # Send sources immediately
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'count': len(retrieved_docs)})}\n\n"

            # Get the context from retrieved docs
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)

            # Create a prompt template
            PROMPT_TEMPLATE = """
            Answer the question based only on the following context:

            {context}

            Question: {question}

            Provide a detailed answer based on the context above.
            Don't justify your answers.
            Don't give information not mentioned in the context.
            Do not say "according to the context" or "mentioned in the context" or similar.
            """

            # Format the prompt
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            formatted_prompt = prompt_template.format_messages(
                context=context,
                question=question
            )

            # Send status update
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"

            # Stream the response from the model
            full_response = ""
            async for chunk in model.astream(formatted_prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

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
