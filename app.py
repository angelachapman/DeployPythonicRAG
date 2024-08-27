import os
from typing import List
import tempfile

import chainlit as cl
from chainlit.types import AskFileResponse
from PyPDF2 import PdfReader

from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings

from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt
)
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.qa_pipeline import RerankedQAPipeline

system_template = """\
Use the following context to answer a users question. If you cannot find the answer in the context, say you don't know the answer."""
system_role_prompt = SystemRolePrompt(system_template)

user_prompt_template = """\
Context:
{context}

Question:
{question}
"""
user_role_prompt = UserRolePrompt(user_prompt_template)

text_splitter = CharacterTextSplitter()
EmbeddingModel = OpenAIEmbeddings()

def process_text_file(file: AskFileResponse):

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file_path = temp_file.name

    with open(temp_file_path, "wb") as f:
        f.write(file.content)

    text_loader = TextFileLoader(temp_file_path)
    documents = text_loader.load_documents()
    texts = text_splitter.split_texts(documents)
    return texts

def process_pdf(file: AskFileResponse) -> list[str]:
    
    # Create a temporary file to store the PDF content
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(file.content)

    # Read the PDF content
    with open(temp_file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Assuming you have a text splitter similar to the one used for text files
    texts = text_splitter.split_texts([text])
    return texts

# Function to build the vector database from a list of texts
async def build_qdrant_vector_database(list_of_text: List[str]) -> Qdrant:
    embeddings = await embedding_model.async_get_embeddings(list_of_text)
    qdrant = Qdrant.from_texts(
        texts=list_of_text,
        embeddings=[embedding.tolist() for embedding in embeddings],
        embedding=embedding_model,
        collection_name="vectors"
    )
    return qdrant

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text or PDF File file to begin!",
            accept=["text/plain","application/pdf"],
            max_size_mb=5,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # load the file
    if "pdf" not in file.name.lower():
        texts = process_text_file(file)
    else: texts = process_pdf(file)

    print(f"Processing {len(texts)} text chunks")

    # Create a dict vector store
    qdrant_db = await build_qdrant_vector_database(texts)
    
    chat_openai = ChatOpenAI()

    # Create a chain
    retrieval_augmented_qa_pipeline = RerankedQAPipeline(
        vector_db_retriever=qdrant_db,
        llm=chat_openai,
        True
    )
    
    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()