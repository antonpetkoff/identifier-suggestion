
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
