import os
import torch as th
import whisper
from time import time
import json
from tenacity import retry, wait_random
from whisper.audio import SAMPLE_RATE
import openai


class Audio2Text:
    def __int__(self, model_size='small'):
        self.device = 'cpu'
        if th.cuda.is_available():
            self.device = 'cuda'
        self.model = whisper.load_model(model_size, device=self.device)

    def speech_to_text(self, audio_path, customer, assignee, sysdate='', systime=''):
        audio = whisper.load_audio(audio_path)
        audio_language = self.get_language(audio)
        if audio_language == 'en':
            res = self.model.transcribe(audio)
        else:
            res = self.model.transcribe(audio)
            res['text'] = self.translate_text(res['text'], orginal_text=audio_language, convert_to='English')

        return {
            'filename': os.path.basename(audio_path),
            'customer': customer,
            'assignee': assignee,
            'date': sysdate,
            'time': systime,
            'content': {
                'text': res['text'],
                'language': audio_language,
                'meta_info': {
                    'audio_segments': len(res['segments']),
                    'duration': audio.shape[0] / SAMPLE_RATE}
            }
        }

    def get_language(self, audio, duration=30):
        clip_audio = whisper.pad_or_trim(audio, length=SAMPLE_RATE * duration)
        result = self.model.transcribe(clip_audio)
        return result['language']

    @staticmethod
    @retry(wait=wait_random(min=5, max=10))
    def translate_text(text, orginal_text='ar', convert_to='en'):
        prompt = f'Translate the following {orginal_text} text to {convert_to}:\n\n{orginal_text}: ' + text + '\n{convert_to}:'
        # Generate response using ChatGPT
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        # Extract the translated English text from the response
        translation = response.choices[0].text.strip()
        return translation




if __name__ == '__main__':
    pass
