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
        self.assignee_name = self.parser.get_records()['assignee_name']
        self.conversation_dict = dict()
        self.whisper_model_name = self.parser.get_audio_attributes()['whisper_model']
        self.device = self.parser.get_audio_attributes()['device']
        self.audio_process = AudioProcess(model_name=self.whisper_model_name,
                                          device=self.device
                                          )
        self.out_path = self.parser.get_records()['out_path']

    def process_records(self):
        self.conversation_dict['data'] = list()
        excel_data = self.read_data(self.parser.get_records()['file_path'])
        audio_records, account_records = self.get_audio_records(excel_data)
        for audio_path, account in tqdm(zip(audio_records, account_records)):
            r = requests.get(audio_path)
            if r.status_code == 200:
                text, duration = self.audio_process.speech_to_text(audio_path)
                text = re.sub("\s\s+", " ", text)
                text = "conversation with Customer {}:{}".format(account, text)
                self.conversation_dict['data'].append({
                    'name': self.assignee_name,
                    'text': text,
                    'customer': account,
                    'date': '',
                    'call duration': duration
                })
            else:
                logging.error("Unable to reach for URL {}".format(audio_path))
                continue
        self.save_record(record=self.conversation_dict, out_path=self.out_path)
        return 0

    def get_audio_records(self, excel_df):
        record_index_col = self.parser.get_records()['audio_path_index']
        client_index_col = self.parser.get_records()['client_index']
        customer_index_col = self.parser.get_records()['customer_index']
        audio_records = excel_df[record_index_col][excel_df[client_index_col] == self.assignee_name].tolist()
        print("Generating Audio records for {}.{} records generated".format(self.assignee_name,len(audio_records)))
        account_records = excel_df[customer_index_col][excel_df[client_index_col] == self.assignee_name].tolist()
        return audio_records, account_records


    @staticmethod
    def read_data(file_path, customer_index_col='Opportunity'):
        if file_path.endswith('.xlsx'):
            excel_df = pd.read_excel(file_path)
            excel_df.dropna(subset=customer_index_col, inplace=True)
            return excel_df

    @staticmethod
    def get_customer_employee_list(xlsx_data,customer_index,employee_index):
        customer_list = xlsx_data[customer_index].unique().tolist()
        employee_list = xlsx_data[employee_index].unique().tolist()
        return customer_list,employee_list

    @staticmethod
    def save_record(record: dict, out_path='tmp.json'):
        with open(out_path, 'w+') as fc:
            json.dump(record, fc, indent=4)
        return 0


if __name__ == '__main__':
    data_orchestrator = DataOrchestrator(cnf_path="/mnt/e/Personal/Samarth/repository/DMAC_ChatGPT/conf/conf.cnf")
    data_orchestrator.process_records()
