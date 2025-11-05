# chat.py - New router for RAG chat
from fastapi import APIRouter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

chat_router = APIRouter(prefix='/api/ml/', tags=['chat'])

#PINECONE 


class ChatRequest(BaseModel):
    user_id: str
    question: str




@chat_router.post("/chat")
async def ask_question(request: ChatRequest):
    try:
        user_id = request.user_id
        question = request.question
        
        # 1. Get user's vectorstore
        hash_name = hashlib.md5(user_id.encode())
        hashed_value = hash_name.hexdigest()
        index_name = f"{hashed_value}-docs"

        #connect to the vectorstore
        pc.connect(index_name=index_name)
        
        # 2. Create retriever
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embed
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 3. Create RAG chain
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # 4. Get answer
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] for doc in result["source_documents"]],
            "question": question
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))