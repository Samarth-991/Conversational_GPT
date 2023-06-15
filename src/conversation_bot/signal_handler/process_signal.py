import configparser
import os


class ConfigParser:
    def __init__(self, file_path):
        self.__config = configparser.ConfigParser()
        if not os.path.isfile(file_path):
            raise "signal config not found "
        self.__config.read(file_path)

    def get_general_attributes(self):
        return dict(self.__config.items('general'))

    def get_vectorstore_attributes(self):
        return dict(self.__config.items('vector_store'))

    def get_audio_attributes(self):
        data = dict(self.__config.items('audio'))
        return data

    def get_streamlit_path(self):
        return dict(self.__config.items('APP'))['streamlit_path']


