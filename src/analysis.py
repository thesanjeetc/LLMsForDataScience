import ast
import re
import os
import json
import pandas as pd
from enum import Enum
from collections import defaultdict

def get_callable_methods(module):
    return [
        m for m in dir(module) 
        if callable(getattr(module, m)) and not m.startswith("_")
    ]

def get_all_callable_pandas_methods():
    return set([c for m in [pd, pd.DataFrame, pd.Series] for c in get_callable_methods(m)])

def __get_pandas_dataframe_return_methods():
    RETURN_TYPE_FILTER = frozenset(["DataFrame", "Series", "NDFrame", "NDFrameT"])
    MISC_ATTRIBUTES = ["DataFrame","Series","iloc", "loc", "ix", "at", "iat", "groupby"]
    res = set(MISC_ATTRIBUTES)
    for module in [pd, pd.DataFrame, pd.Series]:
        methods = get_callable_methods(module)
        for m in methods:
            c = getattr(module, m)
            if hasattr(c, "__annotations__"):
                if "return" in c.__annotations__:
                    return_types = c.__annotations__["return"]
                    for t in RETURN_TYPE_FILTER:
                        if t in str(return_types):
                            res.add(m)
                            break
                elif c.__doc__:
                    texts = [s.strip() for s in str(c.__doc__).split("-") if s.strip()]
                    for i, s in enumerate(texts):
                        if "Returns" in s:
                            first_ret_str = s[i+1].split("\n")[0]
                            for t in RETURN_TYPE_FILTER:
                                if t in first_ret_str:
                                    res.add(m)
                                    break

    return res

def get_pandas_dataframe_return_methods(slice=True):
    with open(os.path.join("resources", "dataframe_return_methods.json")) as f:
        methods = [m["method"] for m in json.loads(f.read())]
    if slice:
        methods.extend(["DataFrame","Series","iloc", "loc", "ix", "at", "iat", "groupby", "filter"])
    return  methods

def get_pandas_dataframe_param_methods():
    PARAM_TYPE_FILTER = frozenset(["DataFrame", "Series", "NDFrame", "NDFrameT"])
    res = set()
    print()
    for module in [pd, pd.DataFrame, pd.Series]:
        methods = get_callable_methods(module)
        for m in methods:
            c = getattr(module, m)
            if hasattr(c, "__annotations__"):
                for k, v in c.__annotations__.items():
                    if k != "return":
                        for t in PARAM_TYPE_FILTER:
                            if t in str(v):
                                res.add(m)
                                break
    return res

DATAFRAME_RETURN_FUNCTIONS = get_pandas_dataframe_return_methods()
DATFRAME_CREATION_FUNCTIONS = [f for f in DATAFRAME_RETURN_FUNCTIONS if f.startswith("read_") or f.startswith("from_")]
DATFRAME_DERIVE_FUNCTIONS = get_pandas_dataframe_param_methods().intersection(DATAFRAME_RETURN_FUNCTIONS)
PANDAS_METHODS = get_all_callable_pandas_methods()

def has_df_creation_call(node):
    result = False
    for _, value in ast.iter_fields(node):
      if isinstance(value, str):
        if value in DATAFRAME_RETURN_FUNCTIONS:
          return True
      elif isinstance(value, list):
        for item in value:
          if isinstance(item, ast.AST):
            result |= has_df_creation_call(item)
          elif isinstance(item, str):
            if item in DATAFRAME_RETURN_FUNCTIONS:
              return True
      elif isinstance(value, ast.AST):
        result |= has_df_creation_call(value)
    return result

def extract_vars(node):
    args = []
    if isinstance(node, ast.Name):
        args.append(node.id)
    elif isinstance(node, ast.Attribute):
        args.append(node.attr)
        args.extend(extract_vars(node.value))
    elif isinstance(node, ast.Tuple):
        for elem in node.elts:
            args.extend(extract_vars(elem))
    elif isinstance(node, ast.BinOp):
        args.extend(extract_vars(node.left))
        args.extend(extract_vars(node.right))
    elif isinstance(node, ast.List):
        for elem in node.elts:
            args.extend(extract_vars(elem))
    elif isinstance(node, ast.Call):
        for arg in node.args:
            args.extend(extract_vars(arg))
        for keyword in node.keywords:
            args.extend(extract_vars(keyword.value))
        args.extend(extract_vars(node.func))
    elif isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            args.extend(extract_vars(key))
            args.extend(extract_vars(value))
    elif isinstance(node, ast.Subscript):
        args.extend(extract_vars(node.value))
        args.extend(extract_vars(node.slice))
    return args

def extract_base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return extract_base_name(node.value)
    elif isinstance(node, ast.Subscript):
        return extract_base_name(node.value)
    return None

class ASTVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.imports = set()
        self.aliases = {}
        self.dataframes = []
        self.dataframe_graph = {}
        self.dataframe_graph_edges = {}

        self.modified_dataframes = []
        self.used_dataframes = []
        self.function_calls = []
        self.new_functions = []
        self.current_vars = []

        self.call_chain = []
        self.current_assign = None
        self.is_current_assign_name = False
        self.current_dataframe_params = []

    def reset(self):
        self.function_calls = []
        self.new_functions = []
        self.modified_dataframes = []
        self.used_dataframes = []
        self.current_vars = []
        self.call_chain = []
        self.current_assign = None
        self.is_current_assign_name = False
        self.current_dataframe_params = []
    
    def add_dataframe(self, name, parent_name, df_args=None, call_chain=None):
        if not call_chain:
            call_chain = [parent_name]
        if df_args:
            for arg in df_args:
                self.dataframe_graph_edges[(arg, name)] = call_chain
        elif parent_name in self.imports or parent_name in self.aliases:
            parent_name = "[SOURCE]"
        self.dataframe_graph_edges[(parent_name, name)] = call_chain
        self.dataframes.append(name)
        self.used_dataframes.append(parent_name)
        self.used_dataframes.extend(df_args if df_args else [])
        if name in set(self.dataframes):
            self.modified_dataframes.append(name)
    
    def visit_Name(self, node: ast.Name):
        if node.id in self.dataframes:
            self.used_dataframes.append(node.id)

    def visit_Assign(self, node: ast.Assign): 
        if len(node.targets) == 1:
            lhs = extract_base_name(node.targets[0])
            if lhs:
                self.is_current_assign_name = isinstance(node.targets[0], ast.Name)
                self.current_assign = lhs
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.dataframes:
                        self.add_dataframe(lhs, node.value.id)
                elif isinstance(node.value, ast.Subscript):
                    if isinstance(node.value.value, ast.Name):
                        if node.value.value.id in self.dataframes:
                            self.add_dataframe(lhs, node.value.value.id)
                else:
                    dfs = [v for v in extract_vars(node.value) if v in self.dataframes]
                    self.current_dataframe_params.extend(dfs)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.asname:
                self.aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.add(f"{node.module}.{alias.name}")
            if alias.asname:
                self.aliases[alias.asname] = f"{node.module}.{alias.name}"
            else:
                self.aliases[alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.new_functions.append(node.name)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        attribute = node.attr
        if isinstance(node.value, ast.Name) or isinstance(node.value, ast.Subscript):
            target = node.value if isinstance(node.value, ast.Name) else node.value.value
            if hasattr(target, "id"):
                module = target.id
                if self.call_chain:
                    self.call_chain.append(module)
                    self.function_calls.append('.'.join(self.call_chain[::-1]))
                    if self.current_assign:
                        if (self.call_chain[0] in DATAFRAME_RETURN_FUNCTIONS or (self.current_assign in set(self.dataframes) and not self.is_current_assign_name)):
                            self.add_dataframe(self.current_assign, self.call_chain[-1], self.current_dataframe_params, self.call_chain)
                            self.current_dataframe_params = []
                            self.current_assign = None
                    self.call_chain = []
                else:
                    self.function_calls.append(f"{module}.{attribute}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if hasattr(node.func, 'attr'):
            self.call_chain.append(node.func.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Attribute) and \
            isinstance(node.value.value, ast.Name):
            self.call_chain.append(node.value.attr)
        self.generic_visit(node)

VISUALISATION_LIBS = frozenset(["matplotlib", "seaborn", "mpl_toolkits", "bokeh", "chartstudio", "folium", "plotly"])


class TaskType(str, Enum):
    VISUAL = "VISUAL"
    TRANSFORM = "TRANSFORM"
    INFO = "INFO"

DATAFRAME_RETURN_METHODS_NO_SLICE = get_pandas_dataframe_return_methods(slice=False)

def generate_metadata(visitor):
    modules = defaultdict(list)
    task_type = None

    for f in visitor.function_calls:
        base, *calls = f.split(".")

        if base in visitor.aliases:
            base = visitor.aliases[base].split(".")[0]
            
        if base in visitor.imports or f in visitor.imports:
            module = base
        elif base in visitor.dataframes or \
            base.startswith("df") or base.endswith("df"):
            module = "pandas"
        else:
            module = "other"
        
        if base in VISUALISATION_LIBS \
            or base == "ax" or base == "fig":
            task_type = TaskType.VISUAL

        modules[module].extend(calls)        
    
    if not task_type:
        if set(modules["pandas"]).intersection(DATAFRAME_RETURN_METHODS_NO_SLICE):
            task_type = TaskType.TRANSFORM
        else:
            task_type = TaskType.INFO

    return {
        "modules": modules,
        "task_type": task_type
    }

def analyse_code(code_context, code):
    visitor = ASTVisitor()
    try:
        ctx_tree = ast.parse(code_context)
        code_tree = ast.parse(code)
        visitor.visit(ctx_tree)
        dataframes_checkpoint = set(visitor.dataframes)
        visitor.reset()
        visitor.visit(code_tree)
        visitor.used_dataframes = [df for df in visitor.used_dataframes if df in dataframes_checkpoint]
        if visitor.call_chain:
            visitor.function_calls.append("." + '.'.join(visitor.call_chain[::-1]))
    except Exception as e:
        print(f"[ANALYSE ERROR]: {e}")
    metadata = generate_metadata(visitor)
    return {
        "imports": visitor.imports,
        "aliases": visitor.aliases,
        "dataframes": visitor.dataframes,
        "used_dataframes": visitor.used_dataframes,
        "modified_dataframes": visitor.modified_dataframes,
        "dataframe_graph": visitor.dataframe_graph,
        "function_calls": visitor.function_calls,
        "new_functions": visitor.new_functions,
        "modules": metadata["modules"],
        "task_type": metadata["task_type"],
    }

def extract_functions(code):
    tree = ast.parse(code)
    functions = {}

    class FunctionExtractor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            start_lineno = node.lineno - 1
            end_lineno = node.end_lineno
            function_code = "\n".join(code.splitlines()[start_lineno:end_lineno])
            functions[node.name] = function_code
            self.generic_visit(node)

    FunctionExtractor().visit(tree)
    return functions