from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil

from pydantic import BaseModel
from typing import List
import uvicorn
from datetime import datetime
import os
import json
import aiofiles

from utils import *
from openai_api import *

app = FastAPI()


try:
    upload_path = './uploaded_files/'
    os.makedirs(upload_path, exist_ok=True)
except:
    print('upload_path exist')

'''
1. 물건, ex 옷.. 찾기 -> yolo ( + 위치설명)
2. 위험 실시간 감지 -> yolo ( + 위험 안내)
3. 옷 색깔이나 디자인 설명, 주변상황 설명, 물건설명, 가게가 어디있는지, 점자인식, 수화인식 -> gpt4o
a. 찾아줘 -> yolo만
b. 길 안내해줘 -> yolo만
c. 설명해줘 -> gpt4o만
'''

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
def find_object(img, txt):
    is_danger = False
    pred = yolo_predict(img, txt)
    if len(pred[0].boxes) > 0:
        results, shp = yolo_processing(pred)
        xmin = results[0]
        ymin = results[1]
        xmax = results[2]
        ymax = results[3]
        height, width = shp[0], shp[1]
        rst_comment = determine_region(xmin, ymin, xmax, ymax, width, height)
        return {'corr': results, 'img_path': img, 'rst_comment' : rst_comment, 'is_concept' : 0}
    else:
        return {'corr': None, 'img_path': None, 'rst_comment' : None, 'is_concept' : 0}
def danger_object(img, txt):
    is_danger = False
    danger_thr = 50
    pred = yolo_predict(img, None)
    if len(pred[0].boxes) > 0:
        results, shp = yolo_danger_processing(pred[0])
        for idx, rst in enumerate(results):
            xmin = rst[0]
            ymin = rst[1]
            xmax = rst[2]
            ymax = rst[3]
            height, width = shp[0], shp[1]
            bounding_box_area = (xmax - xmin) * (ymax - ymin)
            image_area = width * height
            ratio = bounding_box_area / image_area * 100
            if ratio >= danger_thr:
                is_danger = True
                rst_comment = determine_region(xmin, ymin, xmax, ymax, width, height)
            else:
                rst_comment = None
        return {'corr': results, 'img_path' : img, 'rst_comment' : rst_comment, 'is_danger' : is_danger ,'is_concept' : 1}
    else:
        return {'corr': None, 'img_path' : None, 'rst_comment' : None, 'is_danger' : is_danger, 'is_concept' : 1}
    

def background_recog(img, txt):
    # Getting the base64 string
    txt = '시각장애인을 위해 간단하게' + txt
    base64_image = encode_image(img)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": txt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return {'comment': response.json()["choices"][0]["message"]["content"], 'is_concept': 2}


@app.get("/")
def read_root():
    return {"hello": "world"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), text: str = Form(...)):
    if file.content_type != "image/jpeg":
        return JSONResponse(status_code=400, content={"message": "File type not supported"})
    
    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
    
    file_location = f"uploaded_files/{file.filename}-{formatted_now}.jpg"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(file_location)
    print(text)

    text_index, text = text_sim(text)
    print('text_result', text)

    if text_index == 0:
        final_rst = find_object(file_location, text)
    elif text_index == 1:
        final_rst = danger_object(file_location, text)
    elif text_index == 2:
        final_rst = background_recog(file_location, text)
    
    print(final_rst)
    return json.dumps(final_rst)


@app.post("/uploadfile2/")
async def upload_files(mp3_file: UploadFile = File(...), image_file: UploadFile = File(...)):
    if mp3_file.content_type != "audio/webm":
        raise HTTPException(status_code=400, detail="MP3 file type not supported. Please upload an MP3 file.")
    
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Image file type not supported. Please upload a JPEG or PNG file.")
    
    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d%H%M%S") + f"{now.microsecond // 1000:03d}"
    
    image_file_location = f"uploaded_files/{image_file.filename}-{formatted_now}.jpg"
    mp3_file_location = f"uploaded_files/{mp3_file.filename}-{formatted_now}.webm"
    
    try:
        # 이미지 파일 저장
        async with aiofiles.open(image_file_location, "wb") as buffer:
            data = await image_file.read()
            await buffer.write(data)
        
        # WEBM 파일 저장
        async with aiofiles.open(mp3_file_location, "wb") as buffer:
            data = await mp3_file.read()
            await buffer.write(data)

        print(image_file_location)
        print(mp3_file_location)

        text = get_stt(mp3_file_location)

        text_index, text = text_sim(text)
        print('text_result', text)

        if text_index == 0:
            final_rst = find_object(image_file_location, text)
        elif text_index == 1:
            final_rst = danger_object(image_file_location, text)
        elif text_index == 2:
            final_rst = background_recog(image_file_location, text)
        
        print(final_rst)
        return json.dumps(final_rst)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/prompt/")
def prompt_proc():
    
    prompt_string = """
        이 사진에 보이는 물체들에 대해 촬영된 카메라로부터 거리를 계산해주고, 위험 요소가 있는지 없는지 알려줘
    """
    image_path = "test2.jpeg"
    
    result = get_image_result(image_path=image_path, prompt=prompt_string)
    
    return result
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="./selfsigned.key", ssl_certfile="./selfsigned.crt")