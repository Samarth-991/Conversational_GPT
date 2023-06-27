import torch as th
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from googletrans import Translator, constants
import torchaudio
import time


def load_audio(audio_url):
    """Load the audio file & convert to 16,000 sampling rate"""

    speech, sr = torchaudio.load(audio_url)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)
    duration = speech.shape[1] / 16000
    return speech.squeeze(), duration


def run_inference(speech_clip, model, processor, device='cpu'):
    th.cuda.empty_cache()
    model.eval()
    input_features = processor(speech_clip, return_tensors="pt", sampling_rate=16000)["input_values"].to(device)
    logits = model(input_features)["logits"]
    predicted_ids = th.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()


def get_transcription(audio_path, model, processor, device='cpu', n_seconds=30):
    speech, duration = load_audio(audio_path)
    initial_index = 0
    text = []
    if duration>n_seconds:
        for index in range(int(duration / n_seconds)):
            num_samples = initial_index + n_seconds * int((speech.shape[0] / 16000))
            speech_clip = speech[initial_index:num_samples]
            text.append(run_inference(speech_clip, model, processor, device=device))
            initial_index = num_samples
    else:
        return run_inference(speech, model, processor, device=device)

    return " ".join(text)


# arabic model facebook/s2t-wav2vec2-large-en-ar
def speech_to_text(audio_path, model_name='facebook/wav2vec2-base-960h'):
    translator = Translator()
    transcription = dict()
    #device = "cuda" if th.cuda.is_available() else "cpu"
    device = "cpu"
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    transcription['text'] = get_transcription(audio_path, model=wav2vec2_model, processor=wav2vec2_processor, device=device)
    arabic_to_en = translator.translate(transcription['text'],dest='en')
    transcription['translate'] = arabic_to_en.text
    return transcription


if __name__ == '__main__':
    url = "https://prypto-api.aswat.co/surveillance/recordings/44fe5e2b-4428-41a1-826b-5de427b7930e.mp3"
    text = speech_to_text(audio_path=url, model_name='facebook/s2t-wav2vec2-large-en-ar')

    print(text['text'])
    print(text['translate'])
