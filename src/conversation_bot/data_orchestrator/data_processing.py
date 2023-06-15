import json
import requests
from tqdm import tqdm
import pandas as pd
import re
import logging
from src.conversation_bot.utils.audio_to_text import AudioProcess
from src.conversation_bot.signal_handler.process_signal import ConfigParser as Parser


class DataOrchestrator:
    def __init__(self, cnf_path: str):
        self.parser = Parser(cnf_path)
        self.conversation_dict = dict()
        self.whisper_model_name = self.parser.get_audio_attributes()['whisper_model']
        self.device = self.parser.get_audio_attributes()['device']
        self.audio_process = AudioProcess(model_name=self.whisper_model_name,
                                          device=self.device
                                          )
        self.out_path = self.parser.get_records()['out_path']

    def process_records(self, records_list):
        self.conversation_dict['data'] = list()
        for audio_path  in tqdm(records_list):
            r = requests.get(audio_path)
            if r.status_code == 200:
                text, duration, lang, conv_info = self.audio_process.speech_to_text(audio_path)
                text = re.sub("\s\s+", " ", text)
                text = "conversation between Customer {} and sales {} in {} language:{}".format(
                    conv_info['customer_name'], conv_info['representative_name'], lang, text)
                self.conversation_dict['data'].append({
                    'audio_url': audio_path ,
                    'text': text,
                    'customer': conv_info['customer_name'],
                    'relationship_manager':conv_info['representative_name'],
                    'language': lang,
                    'call duration': duration
                })
            else:
                logging.error("Unable to reach for URL {}".format(audio_path))
                continue
        self.save_record(record=self.conversation_dict, out_path=self.out_path)
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
    data_orchestrator = DataOrchestrator(cnf_path="/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/conf/conf.cnf")
    excel_path = "/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/data/Data_Mortgage.xlsx"
    excel_data = pd.read_excel(excel_path)
    excel_data.dropna(subset='Opportunity', inplace=True)
    audio_files = excel_data['Ameyo Recording URL'].to_list()
    data_orchestrator.process_records(audio_files)

