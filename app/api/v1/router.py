from fastapi import APIRouter, File, Request, UploadFile
from app.domain.services import get_message
from app.domain.services import IA
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

        return {"message": IA(image_bytes)}
        # image = tf.image.decode_image(image_bytes)