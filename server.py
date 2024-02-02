from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np

import torch
import base64
import random

app = FastAPI()
templates = Jinja2Templates(directory='templates')

model_path = "/workspaces/codespaces-blank/best.pt"
model = torch.jit.load(model_path, map_location=torch.device('cpu'))

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # para plotar caixas delimitadoras

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse('home.html', {
        "request": request,
    })

@app.get("/drag_and_drop_detect")
def drag_and_drop_detect(request: Request):
    return templates.TemplateResponse('drag_and_drop_detect.html', {
        "request": request,
    })

@app.post("/")
def detect_with_server_side_rendering(request: Request,
                                      file_list: List[UploadFile] = File(...),
                                      img_size: int = Form(640)):

    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                 for file in file_list]

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    results = model(img_batch_rgb, size=img_size)

    json_results = results_to_json(results)

    img_str_list = []
    for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label,
                         color=colors[int(bbox['class'])], line_thickness=3)

        img_str_list.append(base64EncodeImage(img))

    encoded_json_results = str(json_results).replace("'", r"\'").replace('"', r'\"')

    return templates.TemplateResponse('show_results.html', {
        'request': request,
        'bbox_image_data_zipped': zip(img_str_list, json_results),
        'bbox_data_str': encoded_json_results,
    })

@app.post("/detect")
def detect_via_api(request: Request,
                   file_list: List[UploadFile] = File(...),
                   img_size: Optional[int] = Form(640),
                   download_image: Optional[bool] = Form(False)):

    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                 for file in file_list]

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    results = model(img_batch_rgb, size=img_size)
    json_results = results_to_json(results)

    if download_image:
        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
                label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['bbox'], img, label=label,
                             color=colors[int(bbox['class'])], line_thickness=3)

            payload = {'image_base64': base64EncodeImage(img)}
            json_results[idx].append(payload)

    encoded_json_results = str(json_results).replace("'", r'"')
    return encoded_json_results

def results_to_json(results):
    return [
        [
            {
                "class": int(pred[0]),
                "class_name": f"Class_{int(pred[0])}",
                "bbox": [int(x) for x in pred[1:5].tolist()],
                "confidence": float(pred[5]),
            }
            for pred in result
        ]
        for result in results
    ]

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_one_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def base64EncodeImage(img):
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
    return im_b64

if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000)
    opt = parser.parse_args()

    app_str = 'server:app'
    uvicorn.run(app_str, host=opt.host, port=opt.port, reload=True)
