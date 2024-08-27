from typing import List
import tempfile

import chainlit as cl
from chainlit.types import AskFileResponse
import fitz

from langchain_community.embeddings import OpenAIEmbeddings

from aimakerspace.text_utils import CharacterTextSplitter, TextFileLoader
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.qa_pipeline import RerankedQAPipeline

text_splitter = CharacterTextSplitter()
embedding_model = OpenAIEmbeddings()

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
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(file.content)
        temp_file.flush()

    text = ""
    with fitz.open(temp_file_path) as doc:
        for page in doc:
            text += page.get_text().strip()

    text_list = text_splitter.split_texts(text)  
    return text_list

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a Text File file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    # load the file
    texts = process_text_file(file)

    if not texts:
        await cl.Message(content=f"Error: Could not extract any text from input file").send()
    else:
        print(f"Processing {len(texts)} text chunks")

        # Create a dict vector store
        vector_db = VectorDatabase()
        vector_db = await vector_db.abuild_from_list(texts)
        
        chat_openai = ChatOpenAI()

        # Create a chain
        retrieval_augmented_qa_pipeline = RerankedQAPipeline(
            vector_db_retriever=vector_db,
            llm=chat_openai,
        )
        
        # Let the user know that the system is ready
        msg.content = f"Processing `{file.name}` done. You can now ask questions!"
        await msg.update()

        cl.user_session.set("chain", retrieval_augmented_qa_pipeline)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")
    result = await chain.arun_pipeline(message.content,rerank=True)

    async for stream_resp in result["response"]:
        await msg.stream_token(stream_resp)

    await msg.send()