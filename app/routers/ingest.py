from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import asyncio
# In routers/ingest.py, change back to:
from ..services.embeddings import create_chunks_from_pdf
from ..services.vectorstore import create_vector_store

import hashlib

router = APIRouter()


#trigger from node server to ingest the document
#first get the document from the S3 bucket 
#then calling embedding functions to create an embed of the document 
#then calling vectorstore functions to store the document in the vectorstore 
#return success to node server
#send document ID 

#Incoming Pydantic model for body request
class IngestRequest(BaseModel):
    documentId: str
    userId: str
    filePath: str

def process_document_background(document_id: str, user_id: str, file_path: str):
    """Background function to process document"""
    try:
        logging.info(f"Starting background processing for document {document_id}")
        
        # 1. Create chunks from PDF
        logging.info(f"Creating chunks from PDF: {file_path}")
        chunks = create_chunks_from_pdf(file_path)
        logging.info(f"Created {len(chunks)} chunks from document {document_id}")
        
        # 2. Create/update vector store
        hash_name = hashlib.md5(user_id.encode())
        hashed_value = hash_name.hexdigest()
        index_name = f"{hashed_value}-docs"
        
        logging.info(f"Adding chunks to Pinecone index: {index_name}")
        vectorstore = create_vector_store(chunks, index_name)
        
        if vectorstore:
            logging.info(f" Document {document_id} processed successfully - {len(chunks)} chunks added to {index_name}")
        else:
            logging.error(f" Failed to create vectorstore for document {document_id}")
        
    except Exception as e:
        logging.error(f" Background processing failed for document {document_id}: {str(e)}")


@router.post("/ingest")
async def ingest_document(request: IngestRequest, background_tasks: BackgroundTasks):
    try:
        logging.info(f"Processing document {request.documentId} for user {request.userId}")

        # print(f"Document ID: {request.documentId}")
        # print(f"User ID: {request.userId}")
        # print(f"File Path: {request.filePath}")
        
        # # For now, just return success to test the connection
        # return {
        #     "message": "Document received successfully", 
        #     "documentId": request.documentId,
        #     "userId": request.userId,
        #     "filePath": request.filePath
        # }
        
        # # 1. Create chunks from PDF URL
        # chunks = create_chunks_from_pdf(request.filePath)
        
        # # 2. Create/update vector store
        # hash_name = hashlib.md5(request.userId.encode())
        # hashed_value = hash_name.hexdigest() #returns hex value
        # index_name = f"{hashed_value}-docs"  # User-specific index
        # vectorstore = create_vector_store(chunks, index_name)
        
        # if vectorstore:
        #     logging.info(f"Document {request.documentId} processed successfully")
        #     return {"message": "Document ingested successfully", "chunks_count": len(chunks)}
        # else:
        #     raise HTTPException(status_code=500, detail="Failed to create vector store")

        background_tasks.add_task(process_document_background, request.documentId, request.userId, request.filePath)
        
        return {
            "message": "Document queued for processing",
            "documentId": request.documentId,
            "status": "processing",
            "estimated_time": "2-5 minutes"
        }
            
    except Exception as e:
        logging.error(f"Error processing document {request.documentId}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))