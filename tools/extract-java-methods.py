import javalang
import csv
import sys
import itertools
import os

from pipe import *

# cut everything between (including) the body's opening and closing brackets
def get_text_between_positions(text_lines, start_line, start_column, end_line, end_column):
    inner_lines = text_lines[start_line:(end_line - 1)]
    splitted_first_line = text_lines[start_line - 1].split('{')
    first_line_text = '{' + (splitted_first_line[1] if len(splitted_first_line) > 1 else '')
    last_line_text = text_lines[end_line - 1]

    return ' '.join([s.strip() for s in [first_line_text] + inner_lines + [last_line_text]]).strip()

def get_matching_closing_bracket_position(tokens):
    brackets = tokens | where(lambda token: token.value == '{' or token.value == '}')
    started_counting = False
    counter = 0
    end_position = None

    for bracket in brackets:
        if bracket.value == '{':
            started_counting = True
            counter += 1
        elif bracket.value == '}':
            counter -= 1

        if started_counting and counter == 0:
            end_position = bracket.position
            break

    return end_position

def get_method_body(text_lines, tokens, line, column):
    filtered_tokens = tokens \
        | skip_while(lambda token: token.position.line < line) \
        | skip_while(lambda token: token.position.column < column) \
        | skip_while(lambda token: token.value != '{')

    # TODO: can we omit the generator copying
    filtered_tokens, filtered_tokens_copy = itertools.tee(filtered_tokens)
    closing_bracket_position = get_matching_closing_bracket_position(filtered_tokens_copy)

    if closing_bracket_position is None:
        raise ValueError('cannot find closing bracket')

    return get_text_between_positions(
        text_lines,
        start_line=line,
        start_column=column,
        end_line=closing_bracket_position.line,
        end_column=closing_bracket_position.column
    )

def get_type_name(node):
    # isinstance(node.return_type, javalang.tree.BasicType)
    return 'void' if node is None else node.name

def get_arguments(node):
    return list(node.parameters | map(lambda param: [param.name, get_type_name(param.type)]))

def get_attributes(file_name, class_name, text_lines, tokens, method_node):
    if not isinstance(method_node, javalang.tree.MethodDeclaration):
        raise ValueError('The given node is not of type MethodDeclaration')

    node = method_node

    return {
        'file_name': file_name,
        'method_name': node.name,
        'class_name': class_name,
        'body': get_method_body(text_lines, tokens, line=node.position.line, column=node.position.column),
        'return_type': get_type_name(node.return_type),
        'arguments': get_arguments(node),
        'documentation': repr(node.documentation),
        'modifiers': list(node.modifiers),
        'annotations': list(node.annotations | map(lambda annotation: annotation.name)),
    }

def parse_source_code(file_name, source_code_text):
    tree = javalang.parse.parse(source_code_text)
    tokens = list(javalang.tokenizer.tokenize(source_code_text))
    text_lines = source_code_text.splitlines()

    parsed_methods = []

    for t, class_node in tree.filter(javalang.tree.ClassDeclaration):
        class_name = class_node.name
        parsed_methods_in_class = list(class_node.methods \
            | map(lambda tree_node_tuple: get_attributes(file_name, class_name, text_lines, tokens, tree_node_tuple)))
        parsed_methods = parsed_methods + parsed_methods_in_class

    return parsed_methods

fieldnames = [
    'file_name',
    'class_name',
    'method_name',
    'return_type',
    'arguments',
    'body',
    'documentation',
    'modifiers',
    'annotations'
]

def main():
    try:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for file_name in sys.stdin:
            source_code_file_name = file_name.strip()

            with open(source_code_file_name, 'r') as source_code_file:
                try:
                    source_code_text = source_code_file.read()
                    parsed_data = parse_source_code(source_code_file_name, source_code_text)

                    for row in parsed_data:
                        writer.writerow(row)
                except Exception as error:
                    print(f'warn: cannot parse file {source_code_file_name}: {str(error)}', file=sys.stderr)
    except Exception as e:
        print(e.message, file=sys.stderr)
        exit(1)

if __name__ == '__main__':
    main()
