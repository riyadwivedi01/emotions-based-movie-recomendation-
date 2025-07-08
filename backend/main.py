#by Hridayansh, Riya, Ishita, Lokendra
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
import base64
import numpy as np
import cv2
from tmdb import get_movies_by_emotion
import traceback

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str
@app.post("/detect_emotion/")
async def detect_emotion(data: ImageData):
    try:
        img_data = base64.b64decode(data.image.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        print("DeepFace Result:", result)
        emotions = result[0]['emotion']

        # 👇 Custom logic to avoid "neutral" dominating
        strong_emotions = {k: v for k, v in emotions.items() if k != 'neutral' and v > 20}
        if strong_emotions:
            dominant_emotion = max(strong_emotions, key=strong_emotions.get)
        else:
            dominant_emotion = result[0]['dominant_emotion']


 
        # result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        # # print("DeepFace Result:", result)  # 👈 Add this line 
        # dominant_emotion = result[0]['dominant_emotion']

        movies = get_movies_by_emotion(dominant_emotion)
        return {"emotion": dominant_emotion, "movies": movies}


    except Exception as e:
        print("Emotion detection error:", str(e))  # <-- Add this line
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to detect emotion.")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)