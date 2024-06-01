# HelenKeller-Backend

이 Repository는 항해커톤 2024 14팀의 시각장애인 보조 AI 서비스인 '헬렌켈러'의 FastAPI 기반 백엔드 부분입니다.

## Key Features
- 프론트엔드단에서 이미지와 프롬프트를 입력받아 AI 모델 및 API를 이용하여 프롬프트에 맞게 분석 후 결과를 리턴
    - [KoSimCSE-roberta](https://huggingface.co/BM-K/KoSimCSE-roberta) 모델을 이용한 입력 프롬프트 벡터 유사도 계산으로 프롬프트 시나리오 분기 처리
    - 상기한 분기에 맞게 [GPT-4o API](https://platform.openai.com/) 및 [Yolo-world](https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes) 모델을 이용한 투트랙 상황 처리

## Architecture

![Architecture](https://github.com/hanghae-hackathon/Helenkeller-Backend/blob/main/Architecture.jpg)


## Getting started

### Pre-requisites
- Python 3.10.14 이상

### Installation

1. Clone the repository:
```bash
https://github.com/hanghae-hackathon/Helenkeller-Backend.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add Yolo-world AI model pre-trained weight to root folder
```
https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes
```

### Running the application
```bash
uvicorn main:app --reload -host 0.0.0.0 -port 8000
```

## API References

- `POST /uploadfile/` - 프롬프트 및 이미지를 입력받아 AI 모델로 분석 후 결과값을 String 형태로 반환
