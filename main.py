import fitz
import pytesseract
from pdf2image import convert_from_path
import requests
from PIL import Image
import io
import base64

OLLAMA_URL = "http://localhost:11434/api/generate"  # default Ollama endpoint
MODEL = "gemma3:latest"  # multimodal model in Ollama

def extract_text_from_pdf(pdf_path):
    """Extracts text from PDF, using OCR if page is scanned."""
    doc = fitz.open(pdf_path)
    text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()

        if page_text.strip():
            text += page_text
        else:
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            for img in images:
                text += pytesseract.image_to_string(img)

    return text


def query_ollama(prompt, image_path=None):
    """Send prompt (and optionally image) to Ollama (LLaVA)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload["images"] = [img_b64]

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()


def summarize_component(pdf_path):
    text = extract_text_from_pdf(pdf_path)

    prompt = f"""
You are an expert electronics engineer.
Write a ONLY ONE ENGLISH SENTENCE extract of the part described in this document. Focus on most important parameters.
Format the output to be only the necessary data, avoid any extra text. Avoid repetition.
Where possible give data in format: "Parameter ValueUnit" 
Do not add description of component functioning.
At the end, translate the output to english if in other language.
The output CANNOT INCLUDE ANY OTHER LANGUAGES THAN ENGLISH.
Datasheet content:
{text[:4000]}  # keep within context size
"""
    return query_ollama(prompt)


if __name__ == "__main__":
    pdf_file = "examples/datasheet2.pdf"
    summary = summarize_component(pdf_file)
    print(summary)
    out = query_ollama("translate this to english if its in any other language: " + summary)

    print("Summary:", out)

    # Example of also analyzing a diagram image:
    # diagram_summary = query_ollama("Describe the component pinout", image_path="pinout.png")
    # print("Pinout:", diagram_summary)
