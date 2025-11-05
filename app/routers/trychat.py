#Take user input, user id and return the answer

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import hashlib
import os
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
router = APIRouter()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embed = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#takes user Id for hash, question and returns the answer


class ChatRequest(BaseModel):
    userId: str
    question: str
    # Pydantic models for request/response

# class QueryResponse(BaseModel):
#     answer: str
#     sources: List[dict]
#     retrieved_docs: List[str]ß

# class DocumentRequest(BaseModel):
#     text: str
#     metadata: Optional[dict] = {}


@router.post("/chat")
async def create_chat(request: ChatRequest):

    #Take user input, user id and return the answer
    #create a hash of the user_id and connecte the vector database

    print("User ID received", request.userId)
    try:
        #create a hash of the user_id
        hash = hashlib.md5(request.userId.encode()).hexdigest()
        index_name = f"{hash}-docs"

        
        #check if the index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            raise HTTPException(status_code=404, detail="Index not found")

        #get the pinecone index
        index = pc.Index(index_name)

        #create LangChain vectorstore from the pinecone index
        vectorstore = PineconeVectorStore(index=index, embedding=embed)
        print(vectorstore)

        #question 
        question = request.question

        retrieved_docs = vectorstore.similarity_search(question, k=5)

        #TODO: retrieve the source and chunk ID 
        # sources = [doc.metadata for doc in retrieved_docs]
        
        #get the context from retrieve docts
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)



        #create a prompt template
        PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}

        Question: {question}

        Provide a detailed answer based on the context above.
        Don't justify your answers.
        Don't give information not mentioned in the context.
        Do not say "according to the context" or "mentioned in the context" or similar.
        """

        # Create the template once
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


        formatted_prompt = prompt_template.format_messages(context=retrieved_docs, question=question)

        response = model.invoke(formatted_prompt)
        
        return {
            "answer": response.content,
            "sources_count": len(retrieved_docs),
            "context_used": context[:200] + "..." if len(context) > 200 else context
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


