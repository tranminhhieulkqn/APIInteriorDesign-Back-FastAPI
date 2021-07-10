import gc
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from source.Predictor import Predictor

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

predictor = Predictor.getInstance()
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
            output = predictor.ensemble_predict(image_url=url)
            print(predictor)
            end = time.time() - start
            print('time: ', end)
            gc.collect()
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
            # "label": labels[int(np.argmax(output))],
            # "score": np.max(output*100).tolist()
        }
    else:
        return {
            "success": False,
            "message": "File not exist!"
        }
