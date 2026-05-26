# txt_parser.py
from app.parsing.base_parser import BaseParser

def read_txt(file_path):

    """
    Open .txt file and read exactly as written
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


if __name__ == "__main__":
    text = read_txt("example.txt")
    print(text)