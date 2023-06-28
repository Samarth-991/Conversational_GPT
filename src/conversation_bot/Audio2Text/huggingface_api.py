import os
import torch as th
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import re
from googletrans import Translator
import torchaudio
# import huggingface_hub

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# os.environ['OPENAI_API_KEY'] = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
# huggingface_hub.login(token=os.getenv('HUGGINGFACEHUB_API_TOKEN'))


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
    if duration > n_seconds:
        for index in range(int(duration / n_seconds)):
            num_samples = initial_index + n_seconds * int((speech.shape[0] / 16000))
            speech_clip = speech[initial_index:num_samples]

            predicted_text = run_inference(speech_clip, model, processor, device=device)
            arabic_to_en = translate_text(predicted_text)
            print(arabic_to_en)
            text.append(re.sub(r'[^\w]', ' ', arabic_to_en))
            initial_index = num_samples
    else:
        return run_inference(speech, model, processor, device=device)

    return " ".join(text)


def translate_text(text_data, model_name="facebook/nllb-200-distilled-600M", src_lang='arabic'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=True)
    inputs = tokenizer(text_data, return_tensors="pt")
    translated_tokens = model.generate(**inputs,max_length=90,forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text


# arabic model facebook/s2t-wav2vec2-large-en-ar
def speech_to_text(audio_path, model_name='facebook/wav2vec2-base-960h'):
    translator = Translator()
    transcription = dict()
    # device = "cuda" if th.cuda.is_available() else "cpu"
    device = "cpu"
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    transcription['text'] = get_transcription(audio_path, model=wav2vec2_model, processor=wav2vec2_processor,
                                              device=device)
    print(transcription['text'])
    # arabic_to_en = translator.translate(transcription['text'],dest='en')

    transcription['translate'] = arabic_to_en

    return transcription


from tenacity import retry, wait_random

# @retry(wait=wait_random(min=5, max=10))
# def translate_text(self, text, orginal_text='ar', convert_to='english'):
#     prompt = f'Translate the following {orginal_text} text to {convert_to}:\n\n{orginal_text}: ' + text + '\n{convert_to}:'
#     # Generate response using ChatGPT
#     response = openai.Completion.create(
#         engine='text-davinci-003',
#         prompt=prompt,
#         max_tokens=100,
#         n=1,
#         stop=None,
#         temperature=0.7
#     )
#     # Extract the translated English text from the response
#     translation = response.choices[0].text.strip()
#     return translation


if __name__ == '__main__':
    url = "https://prypto-api.aswat.co/surveillance/recordings/44fe5e2b-4428-41a1-826b-5de427b7930e.mp3"
    text = speech_to_text(audio_path=url, model_name='facebook/s2t-wav2vec2-large-en-ar')

    print(text['text'])
    print(text['translate'])
