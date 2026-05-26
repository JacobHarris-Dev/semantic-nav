# parsers/base_parser.py

from abc import ABC, abstractmethod

class BaseParser(ABC):

    @abstractmethod
    def parse(self, file_path):
        """
        Parse a file and return extracted text.
        """
        pass