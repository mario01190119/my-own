# RAG 기반 에이전트

Ollama에 만든 나만의 모델을 PDF RAG와 연결합니다. 같은 근거로 두 모델의 답변을 만든 뒤 검토 모델이 원문과 다시 대조합니다. 모델의 가중치를 학습시키는 활동은 아닙니다.


## 1. 준비

Ollama를 실행하고 이 폴더에서 PowerShell을 엽니다. Python은 3차시에 사용한 환경을 사용합니다.

```powershell
ollama list
python -m pip install -r requirements.txt
```

`bge-m3`와 검토 모델이 목록에 없을 때만 다운로드합니다. 7B 모델 실행이 어려우면 선생님과 검토 모델을 조정합니다. 같은 모델을 두 역할에 넣어 흐름만 체험할 수도 있지만, 이때는 서로 다른 모델의 성능 비교가 아닙니다.

```powershell
ollama pull bge-m3
ollama pull qwen2.5:7b
```

내 모델이 이미 있으면 다시 만들 필요가 없습니다. 다른 PC에서 재현할 때는 학생이 만든 Modelfile을 이 폴더에 복사하고 다음 명령을 실행합니다. `my-slm`은 원하는 모델 이름으로 바꿉니다.

```powershell
ollama create my-slm -f .\Modelfile
```

## 2. 빈칸 채우기

```powershell
notepad settings.py
```

다음 두 빈칸을 채웁니다. 경로를 복사할 때 따옴표가 중복되지 않도록 합니다.

```python
PDF_PATH = r""  # [내 PDF 경로 입력]
MODEL = ""      # [ollama list의 내 모델 이름 입력]
```

PDF는 글자를 선택·복사할 수 있는 약 10페이지 자료를 준비합니다. 스캔 이미지의 OCR 기능은 포함하지 않았습니다. 표·다단 문서의 읽기 순서는 추출 결과를 확인합니다. 경로는 이 폴더 기준 상대 경로도 가능합니다. 인용의 페이지는 인쇄된 쪽 번호가 아니라 PDF 뷰어의 1부터 시작하는 페이지 순서입니다.

## 3. 자료 DB와 화면 실행

```powershell
python build.py
python -m streamlit run 04_rag.py
```

PowerShell에 표시된 Local URL을 브라우저에서 엽니다. 보통 `http://localhost:8501`입니다. 서버 종료는 PowerShell에서 Ctrl+C입니다. PDF·임베딩 모델·조각 크기를 바꾸면 build.py를 다시 실행합니다. TOP_K만 바꾸면 DB를 다시 만들 필요는 없습니다.

## 4. 세 가지 실험

1. **내 모델만**: 자료 없이 기존 모델로 답합니다.
2. **내 모델 + RAG**: 검색된 조각과 페이지를 보고 답합니다.
3. **두 모델 + RAG + 검토**: 내 모델 답변, 검토 모델의 독립 답변, 최종 검토를 순서대로 실행합니다. 생성 요청은 세 번이며 임베딩·모델 정보 조회 요청은 별도입니다.

세 모드에 같은 질문을 직접 다시 입력해 비교합니다. 화면은 이번 질문 한 건만 보여 줍니다. 비교할 답변은 별도로 기록합니다. 추천 질문 유형은 ① 한 페이지의 사실 확인 ② 두 부분 비교 ③ 자료에 없는 내용입니다. 두 부분 비교는 필요한 조각을 모두 찾지 못할 수 있습니다. 검색 결과부터 확인하고 질문이나 TOP_K를 조정합니다.

Modelfile의 SYSTEM을 읽어 자료 사용 규칙과 함께 보냅니다. temperature, num_ctx 등 PARAMETER를 코드에서 덮어쓰지 않습니다. 다만 기존 캐릭터 지시와 자료 근거 지시가 충돌하면 학생이 Modelfile을 조정해야 합니다. 컨텍스트가 작으면 검토 입력이 잘릴 수 있으므로 답변 길이·TOP_K를 줄이거나 모델의 num_ctx를 조정합니다. num_ctx를 바꾸면 필요한 메모리도 늘어납니다.

## 오류 해결

| 상황 | 확인할 것 |
|---|---|
| PDF 경로 입력 안내 | settings.py의 PDF_PATH 빈칸 채우기 |
| 글자를 추출할 수 없음 | 글자가 선택되는 PDF로 바꾸기 |
| 연결 실패 / model not found | Ollama 실행, ollama list, 모델 이름 확인 |
| 자료가 변경되었다는 안내 | python build.py 재실행 |
| 답이 PDF와 다름 | 검색 조각 → 페이지 → 답변 순서로 확인 |
| 검토가 느리거나 메모리 부족 | 단일 모델 + RAG로 먼저 완성, 작은 검토 모델 사용 |
| 조각에 답이 없음 | 질문 표현 또는 TOP_K 조정; 문서에 진짜 없는지 확인 |

## 사용한 공식 문서

- Ollama API: https://docs.ollama.com/api/chat
- Ollama Modelfile: https://docs.ollama.com/modelfile
- pypdf 텍스트 추출: https://pypdf.readthedocs.io/en/stable/user/extract-text.html
- GitHub 파일 업로드: https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

이 프로젝트의 원문 검색과 AI 검토는 정답을 보장하지 않습니다. 최종 판단은 PDF 원문과 직접 대조합니다.
