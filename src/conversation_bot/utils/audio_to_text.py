from tenacity import retry, wait_random
import openai
import torch as th
import whisper
from whisper.audio import SAMPLE_RATE
import logging


class AudioProcess:
    def __init__(self, model_name='small', device='cuda'):
        self.device = device
        if not th.cuda.is_available():
            logging.warning('Cuda Not available, using Device as CPU')
            self.device = 'cpu'
        self.model = whisper.load_model(model_name, device=self.device)  # in config

    def get_language(self, audio, duration=30):
        clip_audio = whisper.pad_or_trim(audio, length=SAMPLE_RATE * duration)
        result = self.model.transcribe(clip_audio)
        return result['language']

    def speech_to_text(self, audio_path):
        try:
            audio = whisper.load_audio(audio_path)
            audio_language = self.get_language(audio)
            if audio_language == 'en':
                res = self.model.transcribe(audio)
            else:
                res = self.model.transcribe(audio)
                res['text'] = self.translate_text(res['text'], orginal_text=audio_language, convert_to='English')
            audio_duration = audio.shape[0] / SAMPLE_RATE
        except IOError as err:
            logging.error("IO error processing {}".format(audio_path,err))
            return '',0.0
        return res['text'], audio_duration

    @retry(wait=wait_random(min=5, max=10))
    def translate_text(self, text, orginal_text='ar', convert_to='en'):
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
