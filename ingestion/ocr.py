import pytesseract
from PIL import Image
import numpy as np

class OCREngine:
    """Performs OCR on images and scanned documents"""
    
    def __init__(self, lang='eng'):
        self.lang = lang
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using Tesseract"""
        try:
            # Preprocess image for better OCR
            img_array = np.array(image.convert('L'))  # Convert to grayscale
            text = pytesseract.image_to_string(img_array, lang=self.lang)
            return text.strip()
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def extract_with_confidence(self, image: Image.Image) -> dict:
        """Extract text with confidence scores"""
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        return {
            'text': ' '.join([word for word in data['text'] if word.strip()]),
            'confidence': np.mean([float(c) for c in data['conf'] if c.isdigit() and float(c) > 0])
        }
