import traceback


def get_method_name():
    stack = traceback.extrack_stack()
    file_name, code_line, func_name, test = stack[-2]
    return func_name
