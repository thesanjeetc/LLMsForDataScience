import os
import ast
import uuid
import json
from tqdm import tqdm
import pickle
import shutil
import types
import gc
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
from src.analysis import ASTVisitor
from IPython.utils.capture import capture_output

np.random.seed(42)

def format_prompt_cells(cells):
    prompt_cells = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            lines = [s.strip() for s in cell["source"].split("#") if s.strip()]
            lines = [f"# {s.strip()}" for l in lines for s in l.split("\n") if s.strip()]
            if prompt_cells and prompt_cells[-1]["cell_type"] == "markdown":
                prompt_cells[-1]["source"] += "\n" + "\n".join(lines)
            else:
                prompt_cells.append({"cell_type": "markdown", "source": "\n".join(lines)})
        else:
            prompt_cells.append(cell)
    return prompt_cells

def generate_schema_and_execution_info(cells, work_dir):
    work_dir = os.path.join(os.getcwd(), "artifacts", work_dir)
    current_dir = os.getcwd()
    tmp_work_dir = os.path.join('/tmp/', str(uuid.uuid4()))
    shutil.copytree(work_dir, tmp_work_dir)
    os.chdir(tmp_work_dir)
    shell = InteractiveShell.instance(user_ns={})
    max_rows = 100
    visitor = ASTVisitor()
    visitor.record_calls = True
    previous_dataframes = set()
    execution_outputs = []
    gc.collect()
    for cell in cells:
        cell_state = {}
        if cell["cell_type"] == "code":
            code = cell["source"]
            code_tree = ast.parse(code)
            visitor.visit(code_tree)
            with capture_output() as captured:
                result = shell.run_cell(code)
            output = result.result if result.success else result.error_before_exec
            cell_state["output"] = None
            cell_state["is_output_dataframe"] = False
            cell_state["output_dataframe_size"] = None
            if not isinstance(output, types.NoneType):
                if isinstance(output, pd.DataFrame) or isinstance(output, pd.Series):
                    cell_state["is_output_dataframe"] = True
                    cell_state["output_dataframe_size"] = len(output)
                    if len(output) > max_rows:
                        output = output.sample(max_rows).sort_index()
                    cell_state["output"] = output.copy()
                else:
                    cell_state["output"] = str(output)
            else:
                if captured.stdout.strip():
                    cell_state["output"] = captured.stdout.strip()
            cell_state["new_dataframes"] = set()
            cell_state["dataframes"] = {}
            cell_state["dataframe_sizes"] = {}
            for k, v in shell.user_ns.items():
                if k in visitor.dataframes and isinstance(v, pd.DataFrame) or isinstance(v, pd.Series):
                    cell_state["dataframe_sizes"][k] = len(v)
                    if len(v) > max_rows:
                        v = v.sample(max_rows).sort_index()
                    cell_state["dataframes"][k] = v.copy()
                    if k in (set(visitor.dataframes) - previous_dataframes):
                        cell_state["new_dataframes"].add(k)
            previous_dataframes |= set(visitor.dataframes)
            cell_state["new_dataframes"] = list(cell_state["new_dataframes"])
        execution_outputs.append(cell_state | cell)
    shell.run_line_magic("reset", "-f")
    os.chdir(current_dir)
    return execution_outputs

def generate_formatted_dataset():
    dataset_path = os.path.join("datasets", "dataset.json")
    output_path = os.path.join("datasets", "dataset.formatted.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    for notebook in dataset:
        for task in notebook["turns"]:
            cells = task["metadata"]["context_cells"]
            task["metadata"]["context_cells_formatted"] = format_prompt_cells(cells)
    with open(output_path, 'w') as f:
        f.write(json.dumps(dataset, indent=4))

def get_execution_state(notebook_name):
    output_name = notebook_name.replace("/", "_")
    output_path = os.path.join("resources", "execution_info", f"{output_name}.pkl")
    with open(output_path, 'rb') as f:
        tasks = pickle.load(f)
    return tasks

def generate_execution_metadata():
    dataset_path = os.path.join("datasets", "dataset.formatted.json")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    for notebook in tqdm(dataset, "Notebooks"):
        output_name = notebook['notebook_name'].replace("/", "_")
        output_path = os.path.join("resources", "execution_info", f"{output_name}.pkl")
        work_dir = notebook["work_dir"]
        tasks = {}
        for i, task in tqdm(enumerate(notebook["turns"]), "Turns"):
            formatted_cells = task["metadata"]["context_cells_formatted"]
            tasks[i] = generate_schema_and_execution_info(formatted_cells, work_dir)
        with open(output_path, 'wb') as f:
            pickle.dump(tasks, f)

if __name__ == "__main__":
    generate_formatted_dataset()

    import matplotlib
    matplotlib.use('Agg')
    generate_execution_metadata()