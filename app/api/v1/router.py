from fastapi import APIRouter, File, Request, UploadFile
from app.domain.services import get_message
from app.domain.services import IA
from app.domain.services2 import IA2
import os
import shutil

import tensorflow as tf

router = APIRouter()

@router.get("/message")
async def message():
    return {"message": get_message()}

@router.post("/ia1")
async def ia1(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file provided"}
    else:
        image_bytes = await file.read()
            
        return {"message": IA2(image_bytes)}
        # image = tf.image.decode_image(image_bytes)