import os
import json
import requests
from tqdm import tqdm
import pandas as pd
import re
import logging

from utils.audio_to_text import AudioProcess
from signal_handler.constant import WHISPER_MODEL,DEVICE,DOCUMENT_OUT_PATH


class DataOrchestrator:
    def __init__(self, ):
        self.conversation_dict = dict()
        self.whisper_model_name = WHISPER_MODEL
        self.device = DEVICE
        self.audio_process = AudioProcess(model_name=self.whisper_model_name,
                                          device=self.device
                                          )
        self.out_path = DOCUMENT_OUT_PATH

    def process_records(self, records_list):
        self.conversation_dict['data'] = list()
        for audio_path in tqdm(records_list):
            if "http://" in audio_path:
                r = requests.get(audio_path)
                if r.status_code == 200:
                    self.process_audio(audio_path)
                else:
                    logging.error("Unable to reach for URL {}".format(audio_path))
                    continue
            else:
                self.process_audio(audio_path)

        self.save_record(record=self.conversation_dict, out_path=self.out_path)
        return self.out_path

    def process_audio(self, audio_path):
        text, duration, lang, conv_info = self.audio_process.speech_to_text(audio_path)
        text = re.sub("\s\s+", " ", text)
        text = "conversation between Customer {} and sales {} in {} language:{}".format(
            conv_info['customer_name'], conv_info['representative_name'], lang, text)
        self.conversation_dict['data'].append({
            'audio_url': audio_path,
            'text': text,
            'customer': conv_info['customer_name'],
            'relationship_manager': conv_info['representative_name'],
            'language': lang,
            'call duration': duration
        })
        return 0

    @staticmethod
    def read_data(file_path, customer_index_col='Opportunity'):
        audio_files = []
        excel_df = pd.read_excel(file_path)
        excel_df.dropna(subset=customer_index_col, inplace=True)
        return excel_df

    @staticmethod
    def save_record(record: dict, out_path='tmp.json'):
        with open(out_path, 'w+') as fc:
            json.dump(record, fc, indent=4)
        return 0


if __name__ == '__main__':
    print(os.getenv('BASE_PATH'))
    #data_orchestrator = DataOrchestrator(cnf_path="/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/conf/conf.cnf")

    #data_orchestrator.process_records(audio_files)
