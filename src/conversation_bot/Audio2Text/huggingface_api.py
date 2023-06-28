import os
import time

import torch as th
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import nltk
import re
import logging
import librosa
import huggingface_hub
import soundfile as sf
import io
from urllib.request import urlopen
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download('punkt')
TMP_PATH = 'tmp.wav'
WAV2VEC_MODEL = 'facebook/s2t-wav2vec2-large-en-ar'
NLLB_MODEL = 'facebook/nllb-200-distilled-600M'

os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
huggingface_hub.login(token=os.getenv('HUGGINGFACEHUB_API_TOKEN'), add_to_git_credential=False)


def load_audio(audio_url):
    """Load the audio file & convert to 16,000 sampling rate"""
    speech, fs = sf.read(io.BytesIO(urlopen(audio_url).read()))
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]
    if fs != 16000:
        speech = librosa.resample(speech, orig_sr=fs, target_sr=16000)
    duration = speech.shape[0] / 16000

    sf.write(TMP_PATH, speech, 16000)
    time.sleep(1)
    return speech, duration


def correct_sentences(input_text):
    words_array = nltk.word_tokenize(input_text)
    return " ".join(words_array)


def run_inference(speech_clip, model, processor, device='cpu'):
    th.cuda.empty_cache()
    model.eval()
    input_features = processor(speech_clip, return_tensors="pt", sampling_rate=16000)["input_values"].to(device)
    logits = model(input_features)["logits"]
    predicted_ids = th.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return " ".join(correct_sentences(transcription))


def get_transcription(audio_path, model, processor, device='cpu', n_seconds=30):
    speech, duration = load_audio(audio_path)
    translated_text = []
    predicted_text = []

    if duration > n_seconds:
        stream = librosa.stream(
            "tmp.wav",
            block_length=n_seconds,
            frame_length=16000,
            hop_length=16000
        )
        for speech_clip in stream:
            if len(speech.shape) > 1:
                speech = speech[:, 0] + speech[:, 1]
            predicted = run_inference(speech_clip, model, processor, device=device)
            arabic_to_en = translate_text(predicted)
            predicted_text.append(re.sub(r'[^\w]', '', predicted))
            translated_text.append(re.sub(r'[^\w]', '', arabic_to_en))
    else:
        predicted = run_inference(speech, model, processor, device=device)
        arabic_to_en = translate_text(predicted)
        return predicted, arabic_to_en
    return " ".join(translated_text), " ".join(predicted_text)


def translate_text(text_data, model_name=NLLB_MODEL, src_lang='arabic'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
    inputs = tokenizer(text_data, return_tensors="pt")
    translated_tokens = model.generate(**inputs, max_length=4096,
                                       forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text


# arabic model facebook/s2t-wav2vec2-large-en-ar
def speech_to_text(audio_path, model_name=WAV2VEC_MODEL):
    logger = create_logger()
    logger.info("Processing {}".format(audio_path))

    transcription = dict()
    # device = "cuda" if th.cuda.is_available() else "cpu"
    device = "cpu"
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    transcription['translated'], transcription['text'] = get_transcription(audio_path, model=wav2vec2_model,
                                                                           processor=wav2vec2_processor,
                                                                           device=device)

    return transcription


def create_logger():
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:- %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("APT_Realignment")
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        logger.addHandler(console_handler)
    logger.propagate = False
    return logger


if __name__ == '__main__':
    url = "https://prypto-api.aswat.co/surveillance/recordings/5a0e3bd9-f603-45b8-a086-8c38be251a73.mp3.mp3"
    text = speech_to_text(audio_path=url, model_name='facebook/s2t-wav2vec2-large-en-ar')
    print(text['translated'])
