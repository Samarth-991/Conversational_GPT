import json
import os

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter

from src.conversation_bot.signal_handler.process_signal import ConfigParser as Parser
from src.conversation_bot.utils.embedder_model import HuggingFaceEmbeddings

"""
from langchain.vectorstores import Pinecone
import pinecone
pinecone.init(
    api_key = os.environ["PINECONE_API_KEY"],
    environment = os.environ["PINECONE_ENVIRONMENT_REGION"]
)
"""


class IngestionToVectorDb:
    def __init__(self, config, chunk_size=500, overlap=0, embedding='huggingface', out_path='vector_db'):
        self.parser = Parser(config)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.data_path = self.parser.get_records()['out_path']
        self.vector_store = self.parser.get_vectorstore_attributes()['local_vector_store']
        self.word_embedder = self.parser.get_vectorstore_attributes()['embedding']

        if self.word_embedder == 'openai':
            from langchain.embeddings.openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings()
        else:
            self.embeddings = HuggingFaceEmbeddings()
        self.conversation_dict = dict()
        self.ingestion()

    def ingestion(self):
        docs = self.data_loader(tmp_path=self.data_path, chunk_size=self.chunk_size, overlap=self.overlap)
        embedding_vector = self.save_to_local_vectorstore(docs, embedding=self.embeddings)
        if embedding_vector:
            embedding_vector.save_local(self.vector_store)

    @staticmethod
    def save_to_local_vectorstore(docs, embedding):
        vectorstore = None
        try:
            from langchain.vectorstores import FAISS
            vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)
        except ImportError as err:
            raise ("{} no module FAISS found. use pip install faiss".format(err))
        return vectorstore

    @staticmethod
    def data_loader(tmp_path, chunk_size=100, overlap=0):
        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata['customer'] = record.get('customer')
            metadata['language'] = record.get('date')
            metadata['duration'] = record.get('call duration')
            return metadata

        loader = JSONLoader(
            file_path=tmp_path,
            jq_schema='.data[]',
            content_key="text",
            metadata_func=metadata_func
        )
        conversation_docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=overlap)
        texts = text_splitter.split_documents(documents=conversation_docs)
        return texts

    @staticmethod
    def save_record(record: dict, out_path='../data/tmp.json'):
        with open(out_path, 'w+') as fc:
            json.dump(record, fc, indent=4)
        return 0


if __name__ == '__main__':
    cnf_path = "/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/conf/conf.cnf"
    IngestionToVectorDb(config=cnf_path)
