import os
import json
import re
import io
import types
import ast
import numpy as np
import pandas as pd
from src.prompt_utils import format_dataframe_str, string_format

from src.utils import from_jsonl

ANALYSE_PROMPT = """
Output an analysis of your answer in the <analysis> tag.
Explain the underlying reasoning that the question is asking for.
Explain whether this is truly reflected in your answer.
Take a deep breath and work on this problem step-by-step.
If you believe your answer is correct, do not make unnecessary changes.
If you believe your answer is incorrect, explain what needs to be changed to ensure correctness.
Output the final code in the <python> tag.
"""

ANALYSE_ERR_PROMPT = """
Your solution was executed in the stateful Jupyter notebook environment.
An error occurred during execution of the code you submitted: 
<error>
{error_text}
</error>

Output an analysis of your answer and the error in the <analysis> tag.
Explain the underlying reasoning that the question is asking for.
Explain whether this is truly reflected in your answer.
Take a deep breath and work on this problem step-by-step.
Explain what needs to be changed to ensure correctness.
Output the final code in the <python> tag.
"""

ANALYSE_ERR_OUTPUT_PROMPT = """
Your solution was executed in the stateful Jupyter notebook environment.
Here are the execution results of the code you submitted:
<results>
{output}
</results>

Output an analysis of your answer in the <analysis> tag.
Explain the underlying reasoning that the question is asking for.
Explain whether this is truly reflected in your answer and results.
Take a deep breath and work on this problem step-by-step.
If you believe your answer is correct, do not make unnecessary changes.
If you believe your answer is incorrect, explain what needs to be changed to ensure correctness.
Output the final code in the <python> tag.
"""

def extract_tables(html_string):
    try:
        return pd.read_html(io.StringIO(html_string))[0]
    except ValueError:
        match = re.search(r'<th>(.*?)</th>', html_string)
        if match:
            column_name = match.group(1)
            return pd.DataFrame(columns=[column_name])
        else:
            return pd.DataFrame()

def parse_outputs(v):
    if not v:
        return v
    if "<code>" in v:
        out = re.findall(r'<code>(.*?)<\/code>', v, re.DOTALL)[0]
        try:
            out = ast.literal_eval(out)
        except (SyntaxError, ValueError):
            pass
        return out
    elif "table" in v:
        return extract_tables(v)

def format_var(v):
    if isinstance(v, pd.DataFrame):
        v = format_dataframe_str(v, max_rows=3)
    elif isinstance(v, str) and "Name:" in v:
        lines = v.strip().split("\n")
        if len(lines) > 7:
            v = "\n".join(lines[:3] + ["..."] + lines[-3:])
    else:
        v = str(string_format(v))
    return v

def format_vars(output_html):
    output_vars = {k: parse_outputs(v) for k, v in output_html.items()}
    prompt = []
    for k, v in output_vars.items():
        if not isinstance(v, types.NoneType):
            if k == "__output__":
                prompt.append("OUTPUT:")
                prompt.append(format_var(v))
            else:
                prompt.append(f"NAME: {k}")
                prompt.append("VALUE:")
                prompt.append(format_var(v))
            prompt.append("\n")
    return "\n".join(prompt).strip()

def generate_analysis_prompt(output, eval_results):
    return [{"role": "assistant", "content": output}, {"role": "user", "content": ANALYSE_PROMPT}]

def generate_analysis_err_prompt(output, eval_results):
    msg = ANALYSE_PROMPT
    if eval_results.get("error_text"):
        msg = ANALYSE_ERR_PROMPT.format(error_text=eval_results["error_text"][:300])     
    return [{"role": "assistant", "content": output}, {"role": "user", "content": msg}]

def generate_analysis_err_out_prompt(output, eval_results):
    msg = ANALYSE_PROMPT
    if eval_results.get("error_text"):
        msg = ANALYSE_ERR_PROMPT.format(error_text=eval_results["error_text"][:300])
    elif eval_results.get("output_html"):
        msg = ANALYSE_ERR_OUTPUT_PROMPT.format(output=format_vars(eval_results["output_html"])[:3000])
    return [{"role": "assistant", "content": output}, {"role": "user", "content": msg}]
