import gc
import os
import time
import math
import numpy as np
import imageio as io
import tflite_runtime.interpreter as tflite

from PIL import Image, ImageOps
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

labels = ['Art Decor', 'Hi-Tech', 'IndoChinese', 'Industrial', 'Scandinavian']

__models = dict({})
__models_dir = 'models/'
__target_size = 224


def __load_models():
    """ Load all model """
    time_start = time.time()

    list_model = [model for model in os.listdir(__models_dir) if ('.tflite' in model)]

    for model_name in list_model:
        # get model path
        model_path = os.path.join(__models_dir, model_name)
        # loading and active model
        model = tflite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        # get name model
        name = str(model_name.split('.tflite')[0])
        # append to array models
        __models[name] = model
    time_end = time.time() - time_start

    print('Load model successfully! Load in: {}'.format(round(time_end, 3)))

def __get_image_from_url(image_url):
    """ Get image from URL """
    # use library skimage to get image
    image = io.imread(image_url)
    # get image and convert color system to RBG
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    return image

def __customize_size(original_size, target_size):
    """"""
    # default ratio = 1
    ratio = 1
    # get size width, height of image
    width, height = original_size
    # get ratio from size of image
    ratio = (width / target_size) if (width < height) else (height / target_size)
    # return new width, height size
    return int(width / ratio), int(height / ratio)

def __get_step_from_size(size):  # size >= target_size
    """ From the target size and size calculate the number of crops """
    # get variable
    target_size = __target_size
    # get frac and whole
    frac, whole = math.modf(size / target_size)
    # if frac > 0.2 => increase whole
    if frac > 0.2:
        whole += 1
    # return result
    return int(whole)

def __crop_image(image, area):
    """ Crop image with  """
    # get variable
    target_size = __target_size
    # crop image with area
    c_img = image.crop(area)
    # return with fit image
    return ImageOps.fit(c_img, (target_size, target_size), Image.ANTIALIAS)

def __soft_voting(output):
    """ Use soft voting for results """
    # return results of soft voting
    return np.sum(output, axis=0) / len(output)

def __data_processing(image):
    """  """
    # get variable
    target_size = __target_size

    # temporary array of images to return
    images = np.empty((0, target_size, target_size, 3), dtype='float32')

    # get size to resize
    w, h = __customize_size(image.size, target_size)

    # resize image
    image = image.resize((w, h))

    # get the number of images that can be taken in rows and columns
    noCol = __get_step_from_size(w)
    noRow = __get_step_from_size(h)

    if noCol == 1 and noRow == 1:  # if can get only 1 image, crop the image in the center

        # get position crop
        x_ = (w - target_size) // 2
        y_ = (h - target_size) // 2

        # crop image
        area = (x_, y_, x_ + target_size, y_ + target_size)
        croped_image = __crop_image(image, area)
        croped_image = np.array(croped_image) / 255
        croped_image = croped_image.reshape(
            1, target_size, target_size, 3).astype(np.float32)

        # add to array
        images = np.append(images, croped_image, axis=0)

    else:  # if can get multi image
        # get step and position max for crop
        x_max, y_max = np.array((w, h)) - target_size  # get max position

        # get step
        stepCol = (x_max // (noCol - 1)) if (noCol > 1) else 1
        stepRow = (y_max // (noRow - 1)) if (noRow > 1) else 1

        # process each image with the found step
        for random_x in range(0, x_max + 1, stepCol):
            for random_y in range(0, y_max + 1, stepRow):
                # crop image
                area = (random_x, random_y, random_x +
                        target_size, random_y + target_size)
                croped_image = __crop_image(image, area)

                # normalize and reshape
                croped_image = np.array(croped_image) / 255
                croped_image = croped_image.reshape(
                    -1, target_size, target_size, 3).astype(np.float32)

                # add to array
                images = np.append(images, croped_image, axis=0)

    # return array
    return images

def __predict(model, images):
    # get variable
    target_size = __target_size

    # get number of images
    noImage = len(images)

    # get input and output of interpreter
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # if input and output not map with input image => reshape
    if noImage != input_details[0]['shape'][0]:
        model.resize_tensor_input(input_details[0]['index'], (noImage, target_size, target_size, 3))
        model.resize_tensor_input(output_details[0]['index'], (noImage, 5))
        model.allocate_tensors()

    # set input images with input layer interpreter
    model.set_tensor(input_details[0]['index'], images)
    # invoke
    model.invoke()
    # get the result in the output layer
    output = model.get_tensor(output_details[0]['index'])

    # soft voting output
    output = __soft_voting(output)

    # return result
    return output

def ensemble_predict(image_url):
    image = __get_image_from_url(image_url)
    images = __data_processing(image=image)
    # images = image.resize((224, 224))
    # croped_image = np.array(images) / 255
    # images = croped_image.reshape(-1, Predictor.__target_size, Predictor.__target_size, 3).astype(np.float32)

    predictions = []
    for model_name in __models:
        prediction = __predict(
            __models[model_name], images)
        predictions.append(prediction)
    predictions = __soft_voting(predictions)
    return predictions


__load_models()

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
            output = ensemble_predict(image_url=url)
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
