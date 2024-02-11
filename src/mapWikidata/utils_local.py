from emoji import UNICODE_EMOJI

chars = set(["-", " ", ':', '(', ')', '.', ',', '{', '}', '?', '!', ';', '\"', "\'", '+', '='])


def is_lat(s:str) -> bool:
    '''
        Checking that the string consists only of Unicode characters and numbers and other common characters
        param:
            s: str for check
        return:
            true if all chars in string - unicode
    '''
    for char in s:
        if char.isalpha() or char.isdigit() and (char not in UNICODE_EMOJI) or char in  chars:
            pass
        else:
            return False
    return True