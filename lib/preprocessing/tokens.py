
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
    except Exception as e:
        # invalid_body_count += 1
        # if invalid_body_count % 1000 == 0:
        #     print(f'invalid_body_count = {invalid_body_count}')
        return []
