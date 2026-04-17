import logging
import pypdf

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using pypdf."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Failed to read PDF at {file_path}: {e}")
        raise RuntimeError(f"Error parsing PDF: {e}")
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a TXT file safely with encoding fallbacks."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Failed to read TXT at {file_path}: {e}")
            raise RuntimeError(f"Error reading text file: {e}")
            
    # If all encodings fail
    logger.error(f"Failed to decode TXT at {file_path} with supported encodings.")
    raise ValueError("Could not decode the text file. Please ensure it is UTF-8 encoded.")
