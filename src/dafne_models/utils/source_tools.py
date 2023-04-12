import ast

def extract_function_source(source_code, function_name):
    """
    Extracts the source code of a specific function from a string containing Python source code.
    Supports nested functions and functions that end with a nested function definition or 'generate_convert' statement.

    Args:
        source_code (str): The Python source code string to extract the function from.
        function_name (str): The name of the function to extract.

    Returns:
        str: The source code string of the requested function, or an empty string if the function was not found.
    """
    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self, function_name):
            self.function_name = function_name
            self.found_function = False
            self.function_source = ""

        def visit_FunctionDef(self, node):
            if node.name == self.function_name:
                self.found_function = True
                self.function_source = ast.unparse(node).strip()
                return
            else:
                self.generic_visit(node)

        def visit_ClassDef(self, node):
            return  # Skip nested class definitions

        def visit_Lambda(self, node):
            return  # Skip lambda functions

    tree = ast.parse(source_code)
    function_visitor = FunctionVisitor(function_name)
    function_visitor.visit(tree)
    if function_visitor.found_function:
        return function_visitor.function_source
    else:
        return ""
