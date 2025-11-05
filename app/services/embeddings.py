#Create an embedding
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

#OPENAI embeddding model


#create a embedding from a pdf
#takes the pdf file as input and returns the chunks

def create_chunks_from_pdf(pdfPath):
    loader = PyPDFLoader(pdfPath)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    #create chunks
    chunks = text_splitter.split_documents(pages)

    #create embeddings
    print(f"Created {len(chunks)} chunks from PDF")
    print(f"First chunk preview: {chunks[0].page_content[:200]}..." if chunks else "No chunks created\n")
    
    #send teh chunks to vectorstore
    return chunks


if __name__ == "__main__":
    aws_pdf_url = 'https://intreli.s3.ap-southeast-1.amazonaws.com/uploads/user_34lUBTwSTiKNSxq1KUS0lLZEKL8/1761986074058-1761985252258-file-example_PDF_500_kB.pdf'
    
    print("Starting PDF processing...")
    print(f"Processing PDF from: {aws_pdf_url}")
    
    try:
        chunks = create_chunks_from_pdf(aws_pdf_url)
        
        # Show sample of chunks
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"Content: {chunk.page_content[:150]}...")
            print(f"Metadata: {chunk.metadata}")
            
    except Exception as e:
        print(f"Error processing PDF: {e}")



    
    