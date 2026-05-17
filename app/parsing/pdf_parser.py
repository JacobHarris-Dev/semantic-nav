# provides functions to extract text from input pdf document

import fitz

# derive text from pdf using fitz. Takes path to pdf file as input
def extract_text_from_pdf(pdf_path):
    
    # open document with provided pdf path
    doc = fitz.open(pdf_path)
    text = ""

    # traverse document and append in-document text to text variable
    for page in doc:
        text += page.get_text()

    # return derived text
    return text
