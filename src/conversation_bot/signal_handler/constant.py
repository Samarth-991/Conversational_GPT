import os
import os.path as osp
from signal_handler.process_signal import ConfigParser as Parser

configparser = Parser("conf.cnf")
## GENERAL CONFIG
BASE_PATH = configparser.get_general_attributes()['base_path']
AUDIO_FILES_DIR = osp.join(BASE_PATH,configparser.get_general_attributes()['audio_path'])
DOCUMENT_OUT_PATH = osp.join(BASE_PATH,configparser.get_general_attributes()['document_output'])
CHAT_HISTORY = configparser.get_general_attributes()['chat_history']

## MODEL ATTRIBUTES
WHISPER_MODEL = configparser.get_audio_attributes()['whisper_model']
DEVICE = configparser.get_audio_attributes()['device']

## Vector Store attributes
VECTOR_STORE_API = configparser.get_vectorstore_attributes()['vector_store_api']
EMBEDDING = configparser.get_vectorstore_attributes()['embedding']
LOCAL_VECTOR_DIR = osp.join(BASE_PATH,configparser.get_vectorstore_attributes()['local_vectorstore_dir'])
LOCAL_VECTOR_DB = osp.join(LOCAL_VECTOR_DIR,configparser.get_vectorstore_attributes()['vector_embedding_index'])

STREAMLIT_PATH = configparser.get_streamlit_path()
