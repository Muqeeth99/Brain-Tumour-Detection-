from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO 
from PIL import Image
import tensorflow as tf
import cv2
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI() 

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# endpoint = 'https://localhost:8501/v1/models/deep-learning:predict'


MODEL = tf.keras.models.load_model("./saved_models/1")
CLASS_NAMES = ["no","yes"]

# @app.get("/ping")
# async def ping():
#     return ("Hello, I am Alive")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))

    return image
# REACT_APP_API_URL=http://0.0.0.0:8000/predict
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img=cv2.resize(image,(256,256))

    img_batch = np.expand_dims(img,0)

    prediction = MODEL.predict(img_batch)

    # json_data={
    #     "instances": img_batch.tolist()
    # }

    # response = requests.post(endpoint,json=json_data)
    # pass

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost',port=8000)
