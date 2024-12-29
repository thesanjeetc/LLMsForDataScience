import os
import re
import types
import json
import pickle
import ast
import pandas as pd
from collections import defaultdict
from src.prompt_templates import *
from src.prompt_utils import *
from src.prompt_templates import DEFAULT_SYSTEM_PROMPT
from src.analysis import ASTVisitor, analyse_code, PANDAS_METHODS, extract_functions

def add_new_dataframe_cells(cells, include_dataframes=None):
    visitor = ASTVisitor()
    visitor.record_calls = True
    last_modified_dataframe = {}
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            code = c["source"]
            tree = ast.parse(code)
            visitor.modified_dataframes = []
            visitor.visit(tree)
            for df in visitor.modified_dataframes:
                last_modified_dataframe[df] = i
    output = []
    last_modified_cells = defaultdict(set)
    for k, v in last_modified_dataframe.items():
        if (not include_dataframes) or (k in include_dataframes):
            last_modified_cells[v].add(k)
    for i, c in enumerate(cells):
        output.append(c)
        if c["cell_type"] == "code":
            if i in last_modified_cells:
                for df in last_modified_cells[i]:
                    cell = {"cell_type": "code", "source": f"print({df})"}
                    cell["output"] =  c["dataframes"][df].copy()
                    cell["output_dataframe_size"] = c['dataframe_sizes'][df]
                    cell["dataframes"] = c["dataframes"]
                    cell["show_dataframe"] = True
                    output.append(cell)
    return output

def generate_cot(cell, code_context):
    code = cell["source"]
    res = analyse_code(code_context, code)
    pandas_methods = list(set([m for m in res["modules"]["pandas"] if m in PANDAS_METHODS]))
    used_dataframes = list(set([df for df in res["used_dataframes"] if df in cell["dataframes"]]))

    pandas_methods_prompt = ""
    dataframes_prompt = ""
    if len(pandas_methods) > 1:
        pandas_methods_prompt = f'# I must use the `{"`, `".join(pandas_methods[:-1])}` and `{pandas_methods[-1]}` methods.'
    elif pandas_methods:
        pandas_methods_prompt = f'# I must use the `{pandas_methods[0]}` method.'
    if len(used_dataframes) > 1:
        dataframes_prompt = f'# I must use `{"`, `".join(used_dataframes[:-1])}` and `{used_dataframes[-1]}`.'
    elif used_dataframes:
        dataframes_prompt = f'# I must use `{used_dataframes[0]}`.'
    cot_prompt = "\n".join(["# Step 1:", dataframes_prompt, pandas_methods_prompt, "\n# Step 2:"]) + "\n"
    return cot_prompt + code

def build_cell_prompts(cells, add_markdown=True, split_into_cells=False, add_outputs=False, add_new_dataframes=False):
    cell_prompts = []
    for i, cell in enumerate(cells):
        prompt = []
        if cell["cell_type"] == "markdown" and add_markdown:
            if split_into_cells:
                prompt.append(f"\n\nCELL ({cell['cell_type'].upper()}):")
                prompt.append("\n".join([l.split("#", 1)[1].strip() if l.startswith("#") else l for l in cell["source"].split("\n")]))
            else:
                prompt.append(f"\n# In[]:")
                prompt.append(cell["source"])
        elif cell["cell_type"] == "code":
            if split_into_cells:
                prompt.append(f"\n\nCELL ({cell['cell_type'].upper()}):")
                prompt.append(f"```python\n{cell['source']}\n```")
            else:
                prompt.append(f"\n# In[]:")
                prompt.append(cell['source'])
            output_prompt = []
            if add_outputs or (add_new_dataframes and cell.get("show_dataframe")):
                if not isinstance(cell["output"], types.NoneType):
                    if isinstance(cell["output"], pd.DataFrame) or isinstance(cell["output"], pd.Series):
                        output_prompt.append("\nEXECUTION RESULTS:" if split_into_cells else f"\n# Out[]:")
                        max_rows = 3 if cell['output_dataframe_size'] > 10 else None
                        output_prompt.append(format_dataframe_str(cell["output"], original_size=cell['output_dataframe_size'], max_rows=max_rows, add_col_types=True))
                    # else:
                    #     output_prompt.append(string_format(str(cell["output"]), 500))
            # if not split_into_cells:
            #     output_prompt = ["\n\n"] + [f"# {l}" for p in output_prompt for l in p.split("\n") if l.strip()]
            #     output_prompt.append("\n")
            prompt.append("\n".join(output_prompt))
            # new_dataframes_prompt = []
            # if add_new_dataframes and cell["new_dataframes"]:
            #     if not split_into_cells:
            #         new_dataframes_prompt.append("NEW DATAFRAMES:")
            #     for k in cell["new_dataframes"]:
            #         df = cell["dataframes"][k]
            #         df_str = format_dataframe_str(df, original_size=cell['dataframe_sizes'][k], max_rows=3)
            #         if split_into_cells:
            #             prompt.append(f"\nCELL (CODE):")
            #             prompt.append(f"```python\n{k}.head()\n```")
            #             prompt.append("EXECUTION RESULTS:")
            #             prompt.append(df_str)
            #         else:
            #             new_dataframes_prompt.append(f"\nNAME: {k}")
            #             name = "DATAFRAME" if isinstance(df, pd.DataFrame) else "SERIES"
            #             new_dataframes_prompt.append(f"{name}:\n{df_str}")
            #     new_dataframes_prompt = ["\n\n"] + [f"# {l}" for p in new_dataframes_prompt for l in p.split("\n")] + ["\n"]
            # prompt.append("\n".join(new_dataframes_prompt))
        cell_prompts.append("\n".join(prompt))
    return cell_prompts

def get_arcade_exemplars_prompt(split_into_cells, typ="steps"):
    with open(os.path.join("resources", "arcade_prompt_exemplars.json")) as f:
        ex = json.loads(f.read())
    ex_idx = [1,3,4,5]
    exemplar_cells = [c for i in ex_idx for c in ex[typ][str(i)]]
    cell_prompts = build_cell_prompts(exemplar_cells, split_into_cells=split_into_cells)
    return "".join(cell_prompts).strip()

def get_current_dataframes(cells_execution_state):
    for cell in cells_execution_state[::-1]:
        if cell["cell_type"] == "code":
            return cell["dataframes"], cell["dataframe_sizes"]

def add_cot_steps(cells_execution_state):
    cells = []
    code_context = []
    for cell in cells_execution_state:
        if cell["cell_type"] == "code":
            code_steps = generate_cot(cell, "\n".join(code_context))
            code_context.append(cell["source"])
            cell["source"] = code_steps
        cells.append(cell)
    return cells

def build_dataframes_prompt(include_dataframes, current_dataframes, current_dataframe_sizes):
    dataframe_prompts = []
    for k in include_dataframes:
        df = current_dataframes[k]
        if not k.startswith("_"):
            dataframe_prompts.append(f"\nNAME: {k}")
            name = "DATAFRAME" if isinstance(df, pd.DataFrame) else "SERIES"
            max_rows = 3 if len(df) > 10 else None
            dataframe_prompts.append(f"{name}:\n{format_dataframe_str(df, original_size=current_dataframe_sizes[k], max_rows=max_rows, add_col_types=True)}")
    return "\n".join(dataframe_prompts)

def get_schemas(include_dataframes, current_dataframes):
    schemas = []
    for k in include_dataframes:
        df = current_dataframes[k]
        if isinstance(df, pd.DataFrame):
            schemas.append({"name": k, "columns": list(df.columns)})
        else:
            schemas.append({"name": k})
    return schemas

def build_dataframes_columns_prompt(include_dataframes, current_dataframes):
    df_col_prompt = []
    df_col_str = defaultdict(list)
    for k in include_dataframes:
        df = current_dataframes[k]
        col_with_types = get_col_names_with_types(df)
        df_col_str[", ".join(col_with_types)].append(k)
    for k, v in df_col_str.items():
        if not k.startswith("_"):
            if k == "":
                df_col_prompt.append(f"SERIES: {', '.join(v)}")
            else:
                df_col_prompt.append(f"DATAFRAMES: {', '.join(v)}\nCOLUMNS: {k}")
    return "\n\n".join(df_col_prompt)

def build_notebook_prompt(
    cells_execution_state, 
    add_markdown=True,
    split_into_cells=False, 
    add_outputs=False, 
    add_new_dataframes=False, 
    append_dataframes=False,
    max_dataframes = 15,
    prioritize_dataframes = True,
    add_exemplars = None,
    append_schema_info = False,
    extract_user_functions = False,
    max_prompt_size=7000,
    max_context_cells=None,
    prompt_template="",
):
    current_dataframes, current_dataframe_sizes = get_current_dataframes(cells_execution_state)
    code = "\n".join([c["source"] for c in cells_execution_state])

    include_dataframes = list(current_dataframes.keys())[-max_dataframes:]
    if prioritize_dataframes and len(current_dataframes) <= max_dataframes:
        include_dataframes = select_dataframes_by_priority(code, current_dataframes.keys(), max_dataframes)

    if add_new_dataframes:
        cells_execution_state = add_new_dataframe_cells(cells_execution_state, include_dataframes)
    
    if add_exemplars == "ast":
        cells_execution_state = add_cot_steps(cells_execution_state)

    cell_prompts = build_cell_prompts(
        cells_execution_state,
        add_markdown,
        split_into_cells, 
        add_outputs, 
        add_new_dataframes
    )

    tokens_left = max_prompt_size - get_num_tokens(prompt_template)

    dataframes_prompt = ""
    if append_dataframes and current_dataframes:
        dataframes_prompt = build_dataframes_prompt(include_dataframes, current_dataframes, current_dataframe_sizes)
        dataframes_prompt_size = get_num_tokens(dataframes_prompt)
        tokens_left -= dataframes_prompt_size

    dataframes_schema_prompt = ""
    if append_schema_info and current_dataframes:
        dataframes_schema_prompt = build_dataframes_columns_prompt(include_dataframes, current_dataframes)
        dataframes_schema_prompt_size = get_num_tokens(dataframes_schema_prompt)
        tokens_left -= dataframes_schema_prompt_size
    
    user_funcs_prompt = ""
    if extract_user_functions:
        funcs = extract_functions(code)
        user_funcs_prompt = "\n\n".join(funcs.values())

    prompt = set()
    for i, (cell_prompt, cell) in enumerate(zip(cell_prompts, cells_execution_state)):
        if cell.get("show_dataframe"):
            prompt.add((i, cell_prompt))
        cell_prompt_size = get_num_tokens(cell_prompt)
        tokens_left -= cell_prompt_size
    if max_context_cells != 0:
        for i, cell_prompt in enumerate(cell_prompts[::-1]):
            if max_context_cells and i >= max_context_cells:
                break
            cell_prompt_size = get_num_tokens(cell_prompt)
            if cell_prompt_size > tokens_left:
                break
            tokens_left -= cell_prompt_size
            prompt.add((i, cell_prompt))
    prompt = [p[1] for p in sorted(list(prompt), key=lambda x: x[0])]
    total_context_cells = len(cell_prompts)
    num_context_cells = len(prompt)
    notebook_context_prompt = '\n'.join(prompt[::-1])

    exemplars_prompt = ""
    if add_exemplars == "arcade":
        exemplars_prompt = get_arcade_exemplars_prompt(split_into_cells).strip()
    elif add_exemplars == "fyp_cot":
        exemplars_prompt = FYP_CoT_EXEMPLARS

    if max_context_cells != 0:
        if num_context_cells != total_context_cells:
            notebook_context_prompt = f"[showing {num_context_cells} out of {total_context_cells} notebook cells]\n\n" + notebook_context_prompt.strip()

    if dataframes_prompt:
        if len(current_dataframes) > max_dataframes:
            dataframes_prompt = "[some dataframes are not shown due to context limit]\n\n" + dataframes_prompt.strip()

    if dataframes_schema_prompt:
        if len(current_dataframes) > max_dataframes:
            dataframes_schema_prompt = "[some dataframes are not listed due to context limit]\n\n" + dataframes_schema_prompt.strip()

    notebook_context_prompt = re.sub(r'\n{3,}', '\n\n', notebook_context_prompt)

    return {
        "notebook_context_prompt": notebook_context_prompt.strip(),
        "exemplars_prompt": exemplars_prompt.strip(),
        "dataframes_prompt": dataframes_prompt.strip(),
        "dataframes_schema_prompt": dataframes_schema_prompt.strip(),
        "user_functions_prompt": user_funcs_prompt.strip(),
        "num_context_cells": num_context_cells,
        "dataframes": get_schemas(include_dataframes, current_dataframes),
        "total_context_cells": total_context_cells
    }

def build_task_prompt(task, cell_execution_info, prompt_config):
    user_intent = task["turn"]["intent"]["value"]
    notebook_prompts = build_notebook_prompt(cell_execution_info, **prompt_config)
    prompt = prompt_config["prompt_template"].format(user_intent=user_intent, **notebook_prompts)
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    return messages, notebook_prompts
