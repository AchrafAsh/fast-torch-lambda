import time

from fastapi import APIRouter
from pydantic import BaseModel

from models.utils import translate

class Data(BaseModel):
    sentence: str

router = APIRouter()

@router.get("/test")
def testing_child_resoure():
    return {"message": "This is a test endpoint!"}

@router.post("/translate")
def translate_sentence(data:Data):
    start_time = time.time()
    translated_sentence = translate(data.sentence)
    end_time = time.time()
    return { "fr": data.sentence, "en": translated_sentence, "duration": end_time - start_time }