import javalang
import re
from itertools import chain

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


# TODO: use a smarter identifier splitter
def get_subtokens(token):
    return split_camel_case(token)


def tokenize_method_body(code):
    # TODO: fix the data, so that it can be parsed in the first place
    try:
        return list(map(lambda token: token.value, javalang.tokenizer.tokenize(code)))
    except Exception:
        # invalid_body_count += 1
        # if invalid_body_count % 1000 == 0:
        #     print(f'invalid_body_count = {invalid_body_count}')
        return []


STRING_LITERAL_TOKEN = '<STR>' # used to mask string literals
MODIFIERS = ['public', 'private', 'protected', 'static']
SUBTOKEN_REGEX = re.compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+ | # Numbers
    .+
''', re.VERBOSE)


def split_subtokens(code_string):
    return [
        subtoken
        for subtoken in SUBTOKEN_REGEX.findall(code_string)
        if not subtoken == '_'
    ]


def tokenize_method(method_body):
    try:
        tokens = list(javalang.tokenizer.tokenize(method_body))
    except:
        print('ERROR in tokenizing: ' + method_body)
        return []

    if len(tokens) == 0:
        return []

    # mask string literals with a special <STR> token
    tokens = map(
        lambda token: STRING_LITERAL_TOKEN if token.value.startswith('"') else token.value,
        tokens
    )

    # split subtokens
    return list(chain.from_iterable([
        split_subtokens(token)
        for token in tokens
        if not token in MODIFIERS # ignore modifiers
    ]))
