from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from Audio_api.whisper_api import WHISPERModel
from Audio_api.huggingface_api import speech_to_text as huggingface_converter


# Models
class RequestBody(BaseModel):
    urls: List[str]


class ResponseItem(BaseModel):
    id: int
    text: str


class ResponseBody(BaseModel):
    data: List[ResponseItem]


# Initialization
whisper_model = WHISPERModel()
api = FastAPI()


# Routes
@api.get("/")
async def root():
    return {"status": "API server is up and running"}


@api.get("/whisper")
async def whisper(req: RequestBody) -> ResponseBody:
    data = []
    for idx, url in enumerate(req.urls):
        text = whisper_model.speech_to_text(url)
        text["id"] = idx + 1
        data.append(text)
    data_dict = {
        "data" : data
    }
    return data_dict


@api.get("/huggingface")
async def huggingface(req: RequestBody) -> ResponseBody:
    data = []
    for idx, url in enumerate(req.urls):
        text = huggingface_converter(audio_path=url, model_name='facebook/s2t-wav2vec2-large-en-ar')
        text["id"] = idx + 1
        data.append(text)
    data_dict = {
        "data" : data
    }
    return data_dict
