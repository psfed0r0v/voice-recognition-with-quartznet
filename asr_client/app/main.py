from fastapi import FastAPI
from fastapi import File, UploadFile, Form

from .inference import get_transcription

app = FastAPI()


@app.post("/upload-voice")
async def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, "wb") as f:
            f.write(contents)
        text = await get_transcription(file.filename)
        return {"text": text}
    except Exception as e:
        return {"message": "There was an error uploading the file", "e": e}
    finally:
        file.file.close()
