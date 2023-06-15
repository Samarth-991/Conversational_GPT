import os
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from typing import Any
from langchain.prompts import PromptTemplate

from utils.embedder_model import HuggingFaceEmbeddings
from typing import List, Dict

os.environ['OPENAI_API_KEY'] = "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"


def create_prompt():
    prompt_template = """
    Analyze conversations between customer and sales person from context.
    If customer is looking for services or property he is potential lead else not interested. 
    Use the context (delimited by <ctx></ctx>) and  chat history (delimited by <hs></hs>)to answer with customer names.
    If you don't know the answer, just say No idea.
    <ctx>
    {context}
    </ctx>
    <hs>
    {chat_history}
    </hs>
    Question: {question}
    Answer stepwise: 
    """
    prompt = PromptTemplate(input_variables=["context", "question","chat_history"], template=prompt_template)
    return prompt


def run_llm(query: str, embedding_model='huggingface', vector_store='', chat_history: List[Dict[str, Any]] = []) -> Any:
    if embedding_model == 'openai':
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.load_local(vector_store, embeddings=embeddings)
    prompt = create_prompt()
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(llm=chat,
                                               retriever=docsearch.as_retriever(),
                                               combine_docs_chain_kwargs={"prompt": prompt},
                                               )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == '__main__':
    vector_path = "/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/data/faiss_dmac_conv"
    run_llm(query="summarize conversations with Hari Kumar very shortly?", vector_store=vector_path)
