
import numpy as np
import string

class kz():
    def __init__(self, standart = "", allowed_liters = string.ascii_letters):
        self.STANDART = self.check_pattern_standart(standart)
        self.ALLOWED_LITERS = allowed_liters
        self.BLACK_LIST = black_list
        self.ALLOWED_NUMBERS = [str(item) for item in np.arange(10)]
        self.REPLACEMENT = {
            "#": {
                "I": "1",
                "Z": "2",#7
                "O": "0",
                "Q": "0",
                "B": "8",
                "D": "0",
                "S": "5",#8
                "T": "7"
            },
            "@": {
                "/": "I",
                "|": "I",
                "¥": "X",
                "€": "C"
            }
        }

    def find(self, text, strong=True):
        text = self.check_is_str(text)
        text = self.delete_all_black_list_characters(text)
        text = text.upper()

        if len(text) < len(self.STANDART):
            return text

        if len(self.STANDART):
            match = self.findFully(text)
            if match :
                return match.group(0)

        if not strong:
            return self.findSimilary(text)
        return text