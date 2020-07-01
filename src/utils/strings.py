def seq_to_camel_case(seq):
    return [seq[0]] + [
        token[0].upper() + token[1:]
        for token in seq[1:]
        if len(token) > 0
    ]
