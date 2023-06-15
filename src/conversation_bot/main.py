import os
import os.path as osp
import logging
from signal_handler.constant import AUDIO_FILES_DIR ,DOCUMENT_OUT_PATH,OPENAPI_KEY
from data_orchestrator.data_processing import DataOrchestrator
from document_ingestion.ingestion import IngestionToVectorDb
from app import app
import argparse

os.environ['OPENAI_API_KEY'] = OPENAPI_KEY

def data_creation():
    audio_files = [osp.join(AUDIO_FILES_DIR,fname) for fname in os.listdir(AUDIO_FILES_DIR)]
    data_orchestrator = DataOrchestrator()
    text_docments = data_orchestrator.process_records(audio_files[:1])
    IngestionToVectorDb()

def main(args):
    if args.data_creation:
        data_creation()
    if args.create_vector():
        if osp.isfile(DOCUMENT_OUT_PATH):
            IngestionToVectorDb()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a conversational bot\nTo create vector embeddings run python main.py --data_creation')
    parser.add_argument('--data_creation',action='store_true',default=None,help='Read Audio data (url/file) and create embeddings')
    parser.add_argument('--create_vector',action='store_true',default=None,help = 'Ingest vector Embeddings')
    parser.add_argument('--run_app',action='store_true',default=None,help='run application')
    parser.add_argument('--openapi_key',default='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',help='required API KEY')
    parser.add_argument('--run',default=None,help='Export STREAMLIT PATH and then run')

    args = parser.parse_args()
    main(args)

