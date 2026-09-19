from pathlib import Path

ROOT = Path(__file__).resolve().parent

# [내 PDF 경로 입력] 따옴표 안에 경로를 붙여 넣으세요.
# 예: PDF_PATH = r"C:\Users\student\Downloads\자료.pdf"
PDF_PATH = r"C:\Users\yoonc\Downloads\4차시_실습파일\4차시_실습파일\[별표 1] 입원환자의 질병군별 질병의 종류(제3조제1항 관련)(상급종합병원의 지정 및 평가 규정).pdf"

# ollama list에서 확인한 '내가 만든 모델' 이름
MODEL = "exaone3.5:2,4b"
# 검토용 모델도 ollama list에 있어야 합니다. PC 사양에 맞게 바꿔도 됩니다.
REVIEW_MODEL = "qwen2.5:7b"
EMBED_MODEL = "bge-m3"
TOP_K = 3
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
