def read_markdown(file_path):
    """
    Open md file and read exactly as written
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text

if __name__ == "__main__":
    markdown_text = read_markdown("README.md")
    print(markdown_text)