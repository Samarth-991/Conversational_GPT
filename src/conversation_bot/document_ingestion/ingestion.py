import json
import os

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from signal_handler.constant import DOCUMENT_OUT_PATH, LOCAL_VECTOR_DB, EMBEDDING, OPENAI_API_KEY
from utils.embedder_model import HuggingFaceEmbeddings
from utils.common import create_logger

log = create_logger
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

"""
from langchain.vectorstores import Pinecone
import pinecone
pinecone.init(
    api_key = os.environ["PINECONE_API_KEY"],
    environment = os.environ["PINECONE_ENVIRONMENT_REGION"]
)
"""


class IngestionToVectorDb:
    def __init__(self, chunk_size=1000, overlap=50, embedding='huggingface', out_path='vector_db'):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.data_path = DOCUMENT_OUT_PATH
        self.vector_store = LOCAL_VECTOR_DB
        self.word_embedder = EMBEDDING
        log.info("Parameter list ..\nchunk Size:{}\nOverlap:{}\nEmbedder:{}".format(self.chunk_size, self.overlap,
                                                                                    self.word_embedder))
        if not os.path.isfile(self.data_path):
            raise ("error in reading documents data..")

        if self.word_embedder == 'openai':
            log.info("Using Open AI vector embeddings")
            self.embeddings = OpenAIEmbeddings(chunk_size=1000)
        else:
            log.info("Using Hugging face vector embeddings")
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
            vectorstore = FAISS.from_documents(documents=docs, embedding=embedding, )
        except ImportError as err:
            raise ("{} no module FAISS found. use pip install faiss".format(err))
        return vectorstore

    @staticmethod
    def data_loader(tmp_path, chunk_size=1000, overlap=0):
        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata['customer'] = record.get('customer')
            metadata['language'] = record.get('language')
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
