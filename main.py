import gc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from source.Predictor import Predictor

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

labels = ['Art Decor', 'Hi-Tech', 'IndoChinese', 'Industrial', 'Scandinavian']


@app.get("/")
async def root():
    return {"message": "Hello World"}


class PredictBody(BaseModel):
    url: str

@app.post("/predict")
async def predict(body: PredictBody):
    url = body.url
    if url is not None:
        try:
            import time
            start = time.time()
            predictor = Predictor.getInstance()
            output = predictor.ensemble_predict(image_url=url)
            print(predictor)
            del predictor
            gc.collect()
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            end = time.time() - start
            print('time: ', end)
        except():
            return {
                "success": False,
                "message": "File not exist!"
            }

        return {
            "success": True,
            "message": "Predicted Results",
            "result": (output*100).tolist(),
            "Predicted time": str(round(end, 2)),
            "label": labels[int(np.argmax(output))],
            "score": np.max(output*100).tolist()
        }
    else:
        return {
            "success": False,
            "message": "File not exist!"
        }
