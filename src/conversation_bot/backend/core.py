import os
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from typing import Any
from langchain.prompts import PromptTemplate
from signal_handler.constant import EMBEDDING
from langchain.embeddings.openai import OpenAIEmbeddings
from utils.embedder_model import HuggingFaceEmbeddings
from typing import List, Dict

os.environ['OPENAI_API_KEY'] = "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"


def create_prompt():
    prompt_template = """
    Analyze conversations from context.
    If customer is looking for services or property he can be potential lead. 
    Use the context with chat history to answer with customer names.Don't make up answers
   
    {context}
   
    {chat_history}
   
    Question: {question}
    Answer stepwise: 
    """
    prompt = PromptTemplate(input_variables=["context", "question","chat_history"], template=prompt_template)
    return prompt


def run_llm(query: str, embedding_model='openai', vector_store='', chat_history: List[Dict[str, Any]] = []) -> Any:

    if embedding_model == 'open_ai':
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.load_local(vector_store, embeddings=embeddings)
    prompt = create_prompt()
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(llm=chat,
                                               retriever=docsearch.as_retriever(),
                                               combine_docs_chain_kwargs={"prompt": prompt},
                                               max_tokens_limit=4097
                                               )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == '__main__':
    vector_path = "/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/data/faiss_dmac_conv"
    run_llm(query="summarize conversations with Hari Kumar very shortly?", vector_store=vector_path)
