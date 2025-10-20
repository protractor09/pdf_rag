from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf(file_path: str):
    """
    Extract text from PDF.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks for embedding.
    """
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", ".", "!", "?", " "]
    )
    chunks= splitter.split_text(text)
    return chunks




