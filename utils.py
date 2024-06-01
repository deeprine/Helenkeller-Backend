from ultralytics import YOLOWorld
from googletrans import Translator
from faster_whisper import WhisperModel

import torch
from transformers import AutoModel, AutoTokenizer


model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta')
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')

model_size = "large-v3"
model_stt = WhisperModel(model_size, device="cuda", compute_type="float16")

translator = Translator()
yolo_model = YOLOWorld("yolov8x-worldv2.pt")

def determine_region(xmin, ymin, xmax, ymax, img_width, img_height):
    # 이미지의 중앙 영역 크기 정의 (예: 전체 크기의 1/4)
    central_width = img_width / 4
    central_height = img_height / 4
    # 이미지의 중앙 좌표 정의
    center_x = img_width / 2
    center_y = img_height / 2
    # xmin, ymin, xmax, ymax 중심 좌표 계산
    box_center_x = (xmin + xmax) / 2
    box_center_y = (ymin + ymax) / 2
    # 중앙 영역 체크
    if (center_x - central_width / 2 < box_center_x < center_x + central_width / 2) and \
            (center_y - central_height / 2 < box_center_y < center_y + central_height / 2):
        return '중앙'
    # 4분할 영역 체크
    if box_center_x < img_width / 2 and box_center_y < img_height / 2:
        return '왼쪽 상단'
    elif box_center_x >= img_width / 2 and box_center_y < img_height / 2:
        return '오른쪽 상단'
    elif box_center_x < img_width / 2 and box_center_y >= img_height / 2:
        return '왼쪽 하단'
    elif box_center_x >= img_width / 2 and box_center_y >= img_height / 2:
        return '오른쪽 하단'
    return '알 수 없음'

def get_stt(audio_file):
    segments, info = model_stt.transcribe(audio_file, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    for segment in segments:
        text = segment.text
    text = text.strip()
    return text

def cal_score(a, b):
    if len(a.shape) == 1: a = a.unsqueeze(0)
    if len(b.shape) == 1: b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100

def text_sim(text):
    find_keywords = '찾아줘'
    danger_keywords = '안내해줘'
    explan_keywords = '설명해줘'

    if text == find_keywords:
        res = translator.translate(text, src='ko', dest='en')
        text = res.text
        return 0, text

    elif text == danger_keywords:
        res = translator.translate(text, src='ko', dest='en')
        text = res.text
        return 1, text

    elif text == explan_keywords:
        return 2, text

    else:
        inputs = tokenizer([text, find_keywords, danger_keywords, explan_keywords], padding=True, truncation=True, return_tensors="pt")
        embeddings, _ = model(**inputs, return_dict=False)

        find_socre = cal_score(embeddings[0][0], embeddings[1][0])[0].cpu().detach().numpy().tolist()[0]
        danger_score = cal_score(embeddings[0][0], embeddings[2][0])[0].cpu().detach().numpy().tolist()[0]
        explan_score = cal_score(embeddings[0][0], embeddings[3][0])[0].cpu().detach().numpy().tolist()[0]

        result_list = [find_socre, danger_score, explan_score]
        max_val = max(result_list)
        max_index = result_list.index(max_val)

        res = translator.translate(text, src='ko', dest='en')
        text = res.text

        return max_index, text

def yolo_processing(results):
    result = results[0]
    h, w = result.orig_shape[0], result.orig_shape[1]
    boxes = result.boxes
    xyxy = boxes.xyxy[0].cpu().detach().numpy().tolist()

    xmin = xyxy[0]
    ymin = xyxy[1]
    xmax = xyxy[2]
    ymax = xyxy[3]

    return [xmin, ymin, xmax, ymax], [h, w]

def yolo_danger_processing(results):
    xyxy_list = []
    for rst in results:
        print('aa')
        h, w = rst.orig_shape[0], rst.orig_shape[1]
        boxes = rst.boxes
        xyxy = boxes.xyxy[0].cpu().detach().numpy().tolist()

        xmin = xyxy[0]
        ymin = xyxy[1]
        xmax = xyxy[2]
        ymax = xyxy[3]

        xyxy_list.append([xmin, ymin, xmax, ymax])

    return xyxy_list, [h, w]

def yolo_predict(image_path, txt):
    if txt is not None:
        yolo_model.set_classes([txt])

    results = yolo_model.predict(image_path)

    return results

