import javalang
import re
import numpy as np

from pydash import flatten

"""

print(split_camel_case('transformSearchResponse'))
> ['transform', 'search', 'response']

print(split_camel_case('modifySearchResponseData'))
> ['modify', 'search', 'response', 'data']

"""
def split_camel_case(str):
    words = [[str[0]]]

    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return [''.join(word).lower() for word in words]


def get_subtokens(token):
    return split_camel_case(token)


def tokenize_method_body(code):
    # TODO: fix the data, so that it can be parsed in the first place
    try:
        return list(map(lambda token: token.value, javalang.tokenizer.tokenize(code)))
    except Exception:
        return []


STRING_LITERAL_TOKEN = '<str>' # used to mask string literals
NUMBER_LITERAL_TOKEN = '<num>' # user to mask number literals

SUBTOKEN_REGEX = re.compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+ # Numbers
''', re.VERBOSE)


def split_subtokens(code_string, mask_numbers):
    return [
        NUMBER_LITERAL_TOKEN if mask_numbers and re.search(r'^\d+$', subtoken) else subtoken
        for subtoken in SUBTOKEN_REGEX.findall(code_string)
        if subtoken != ''
    ]


def process_token(token, mask_strings, mask_numbers):
    if mask_strings and type(token) == javalang.tokenizer.String:
        return STRING_LITERAL_TOKEN
    elif type(token) == javalang.tokenizer.Identifier:
        return split_subtokens(token.value, mask_numbers)
    return token.value


def tokenize_method(method_body, lowercase=False, mask_strings=True, mask_numbers=True):
    try:
        tokens = list(javalang.tokenizer.tokenize(method_body))
    except:
        print('ERROR in tokenizing: ' + method_body)
        return []

    if len(tokens) == 0:
        return []

    # split subtokens of identifiers
    processed_tokens = flatten([
        process_token(token, mask_strings, mask_numbers)
        for token in tokens
    ])

    if lowercase:
        return np.asarray([token.lower() for token in processed_tokens])

    return np.asarray(processed_tokens)
