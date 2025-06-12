import requests
from bs4 import BeautifulSoup
from uuid import uuid4

def clean_text(text: str) -> str:
    """Collapse whitespace and trim."""
    return ' '.join(text.strip().split())

def split_into_chunks(text: str, max_length: int = 500) -> list[str]:
    """
    Break text into chunks up to max_length characters,
    splitting on sentence boundaries where possible.
    """
    sentences = text.split('. ')
    chunks, current = [], ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # re-append the period we split on
        piece = sentence + ('.' if not sentence.endswith('.') else '')
        if len(current) + len(piece) <= max_length:
            current += piece + " "
        else:
            chunks.append(current.strip())
            current = piece + " "
    if current:
        chunks.append(current.strip())
    return chunks

# … your clean_text() and split_into_chunks() as before …

def upload_url_to_chroma(url: str, user_id: str, collection) -> dict:
    """
    Fetches a web page with a browser-like User-Agent, extracts text, chunks it,
    and stores in the given ChromaDB collection.
    """
    # 1) Prepare headers to avoid basic bot-blocking
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }

    try:
        # 2) Fetch with headers
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        # 3) Parse & clean
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        raw = soup.get_text(separator=' ', strip=True)
        text = clean_text(raw)
        if not text:
            return {"status": "no_content", "chunk_count": 0, "previews": []}

        # 4) Chunk
        chunks = split_into_chunks(text, max_length=500)
        if not chunks:
            return {"status": "no_chunks", "chunk_count": 0, "previews": []}

        # 5) Prepare IDs & metadata
        ids = [f"{user_id}_{uuid4().hex}" for _ in chunks]
        metadatas = [{"source": url, "user_id": user_id, "chunk_index": i}
                     for i in range(len(chunks))]

        # 6) Store
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)

        # 7) Return success with previews
        previews = [c[:200] for c in chunks]
        return {"status": "success", "chunk_count": len(chunks), "previews": previews}

    except requests.HTTPError as http_err:
        # If it's a 403 or other status, report it cleanly
        return {
            "status": "http_error",
            "chunk_count": 0,
            "error": f"{http_err.response.status_code} {http_err.response.reason}"
        }

    except Exception as e:
        # Any other failure
        return {"status": "error", "chunk_count": 0, "error": str(e)}