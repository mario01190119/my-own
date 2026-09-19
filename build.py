"""3차시 build.py 확장: PDF → 페이지별 조각 → 임베딩 → Chroma."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from settings import ROOT, PDF_PATH, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(pdf_path):
    if not str(pdf_path).strip():
        raise ValueError('settings.py의 PDF_PATH = r""에 [내 PDF 경로 입력]을 완료하세요.')
    path = Path(pdf_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("PDF 파일 경로를 확인하세요.")
    reader = PdfReader(path)
    pages, empty = [], []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            pages.append(Document(page_content=text,
                                  metadata={"source": path.name, "page": number}))
        else:
            empty.append(number)
    if not pages:
        raise ValueError("추출된 글자가 없습니다. 글자를 선택할 수 있는 PDF를 사용하세요.")
    return path, pages, empty


def main():
    path, pages, empty = load_pdf(PDF_PATH)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(pages)
    for i, doc in enumerate(docs, start=1):
        doc.metadata["chunk_id"] = i
    print(f"읽은 페이지: {len(pages)} / 만든 조각: {len(docs)}")
    if empty:
        print(f"주의: 글자가 없는 페이지 {empty}는 검색 대상에서 제외했습니다.")
    print("첫 조각 확인:\n", docs[0].page_content[:350])
    db_dir = "chroma_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    Chroma.from_documents(
        documents=docs, embedding=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name="class_rag", persist_directory=str(ROOT / db_dir),
        ids=[str(i) for i in range(len(docs))],
    )
    manifest = {
        "db_dir": db_dir, "embed_model": EMBED_MODEL,
        "collection": "class_rag", "chunk_count": len(docs),
        "source": path.name,
        "pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
    }
    # 새 DB가 완성된 뒤에만 사용 중인 DB 표시를 바꿉니다.
    (ROOT / "active_db.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("완료! python -m streamlit run 04_rag.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"자료 준비 실패: {error}")
