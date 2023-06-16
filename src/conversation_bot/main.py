import logging
import os
import os.path as osp
from signal_handler.constant import AUDIO_FILES_DIR, DOCUMENT_OUT_PATH, STREAMLIT_PATH
import argparse


def data_creation():
    try:
        from data_orchestrator.data_processing import DataOrchestrator
        audio_files = [osp.join(AUDIO_FILES_DIR, fname) for fname in os.listdir(AUDIO_FILES_DIR) if
                       fname.endswith('.mp3')]
        data_orchestrator = DataOrchestrator()
        text_docments = data_orchestrator.process_records(audio_files)
    except ImportError as err:
        logging.error("Import error :{}".format(err))


def main(args):
    if args.create_data:
        data_creation()

    if args.create_vector:
        try:
            from document_ingestion.ingestion import IngestionToVectorDb
            if osp.isfile(DOCUMENT_OUT_PATH):
                print("creating vector from file {}".format(DOCUMENT_OUT_PATH))
                IngestionToVectorDb()
        except ImportError as err:
            logging.error("Import error :{}".format(err))

    if args.app:
        os.environ["STREAMLIT_PATH"] = STREAMLIT_PATH
        os.system("{} run app.py --server.port {}".format(STREAMLIT_PATH,args.port))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run a conversational bot\nTo create vector embeddings run python main.py --data_creation')
    parser.add_argument('--create_data', action='store_true', default=None,
                        help='Read Audio data (url/file) and create embeddings')
    parser.add_argument('--create_vector', action='store_true', default=None, help='Ingest vector Embeddings')
    parser.add_argument('--app', action='store_true', default=None, help='run application')
    parser.add_argument('--openapi_key', default=None, help='required API KEY')
    parser.add_argument('--port',default='8080',help="port number ")

    args = parser.parse_args()
    main(args)
