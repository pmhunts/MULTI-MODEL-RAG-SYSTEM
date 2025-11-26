import PyPDF2
import pdfplumber
from PIL import Image
import io
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DocumentElement:
    """Represents a single element from a document"""
    type: str  # 'text', 'table', 'image'
    content: Any
    page_num: int
    metadata: Dict[str, Any]

class MultiModalParser:
    """Parses PDFs and extracts text, tables, and images"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def parse_document(self, file_path: str) -> List[DocumentElement]:
        """Main parsing function that orchestrates extraction"""
        elements = []
        
        # Extract text and tables using pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    elements.append(DocumentElement(
                        type='text',
                        content=text,
                        page_num=page_num,
                        metadata={'source': 'pdfplumber'}
                    ))
                
                # Extract tables
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        elements.append(DocumentElement(
                            type='table',
                            content=table,
                            page_num=page_num,
                            metadata={'table_idx': table_idx}
                        ))
        
        # Extract images using PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                if '/XObject' in page['/Resources']:
                    xObject = page['/Resources']['/XObject'].get_object()
                    
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            try:
                                img_data = xObject[obj].get_data()
                                img = Image.open(io.BytesIO(img_data))
                                
                                elements.append(DocumentElement(
                                    type='image',
                                    content=img,
                                    page_num=page_num,
                                    metadata={'format': img.format}
                                ))
                            except:
                                continue
        
        return elements
