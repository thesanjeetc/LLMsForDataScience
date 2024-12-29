import os
import re
import ast
import math
import json
import types
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from src.analysis import ASTVisitor
from models.llama3.tokenizer import Tokenizer

tokenizer = Tokenizer("models/llama3/tokenizer.model")

def get_num_tokens(s):
    return len(tokenizer.encode(s, bos=False, eos=False))

def get_type(s):
    if isinstance(s, pd.DataFrame):
        return type(s).__name__
    elif isinstance(s, pd.Series):
        if s.dtype != "object" or len(s) == 0:
            return s.dtype
        else:
            if len(s) > 0:
                return type(s.iloc[0]).__name__
    else:
        return type(s).__name__

def get_col_names_with_types(df):
    series = [(col, df[col]) for col in df.columns] if isinstance(df, pd.DataFrame) else [(df.name, df)]
    return [f"{n} ({get_type(c)})" for n, c in series]

def string_format(s, max_chars = 50):
    if isinstance(s, str):
        if len(s) > max_chars:
            s = s[:max_chars // 2] + "..." + s[-(max_chars // 2):]
    elif isinstance(s, (set, list)):
        s = string_format(str(s), max_chars = 100)
    elif isinstance(s, tuple):
        s = tuple(string_format(item, max_chars = 50) for item in s)
    elif isinstance(s, dict):
        for k, v in s.items():
            s[k] = string_format(v, max_chars = 20)
    elif isinstance(s, float):
        s = "{:.2f}".format(s)
    return s

def format_dataframe_cells(df):
    if len(df) > 0:
        if isinstance(df, pd.DataFrame):
            df = df.round(2)
            for col in df.select_dtypes(include=['string','object']).columns:
                if get_type(df[col]) == "str":
                    df[col] = df[col].apply(lambda s: f"'{s}'")
                df[col] = df[col].apply(string_format)
        else:
            if df.dtype == 'float64':
                df = df.apply(lambda x: "{:.2f}".format(x))
            elif df.dtype in ["string", "object"]:
                df = df.apply(string_format)
    return df

def format_dataframe_str(df, original_size=None, max_rows=20, max_tokens=250, add_col_types=False):
    df = format_dataframe_cells(df)
    if add_col_types:
        df.columns = get_col_names_with_types(df)
    if max_rows == 0:
        df_output = df.drop(df.index)
    elif max_rows and len(df) > max_rows:
        df_output = df.sample(max_rows, random_state=123).sort_index()
    else:
        df_output = df
    num_tokens = get_num_tokens(df_output.to_string())
    if max_tokens and num_tokens > max_tokens:
        tot_tokens = 0
        df_iterrows = df_output.iterrows() if isinstance(df_output, pd.DataFrame) else df_output.items()
        df_rows = []
        for i, row in df_iterrows:
            token_count = get_num_tokens(str(row))
            if tot_tokens + token_count > max_tokens:
                break
            tot_tokens += token_count
            df_rows.append(i)
        df_output = df_output.loc[df_rows]
    original_size = original_size if original_size else len(df)
    df_str = df_output.to_string()
    if len(df_output) == 0 and original_size > 0:
        df_output =  df.head(1)
        df_str = df_output.to_string(max_colwidth=10)
        if get_num_tokens(df_str) > max_tokens:
            omit_msg = "[omitted due to size]"
            df_str = f"Columns: {string_format(list(df.columns), 150)}"
            if get_num_tokens(df_str) > max_tokens:
                df_str = omit_msg
            else:
                df_str += f"\n{omit_msg}"
    if len(df_output) < original_size:
        df_str = f"[showing {len(df_output)} sample rows out of {original_size} rows]\n" + df_str
    return df_str

COMPLEX_DATAFRAME_METHODS = frozenset({'aggregate', 'crosstab', 'pivot_table', 'cut', 'concat', 'qcut', 'merge', 'join', 'agg', 'melt', 'groupby'})

import networkx as nx
import matplotlib.pyplot as plt

def display_dataframe_graph(G, base_dfs, complex_dfs, selected_dfs):
    G = G.copy()
    G.remove_node("[SOURCE]")
    layout = nx.circular_layout(G, scale=3)
    plt.figure(figsize=(5, 4))
    nx.draw_networkx_edges(G, pos=layout, edge_color='#666666', width=1, alpha=0.6,
                           arrowsize=6,  connectionstyle='arc3,rad=0.1')
    node_sizes = {node: 300 if node in selected_dfs else 200 for node in G.nodes()}
    nx.draw_networkx_nodes(G, pos=layout, node_size=[node_sizes[node] for node in G.nodes()], node_color='#E0E0E0', alpha=0.9)
    nx.draw_networkx_nodes(G, pos=layout, nodelist=selected_dfs, node_color='#FFA500', node_size=300, alpha=0.9)
    nx.draw_networkx_nodes(G, pos=layout, nodelist=complex_dfs, node_color='#4ECDC4', node_size=300, alpha=0.9)
    nx.draw_networkx_nodes(G, pos=layout, nodelist=base_dfs, node_color='#FF6B6B', node_size=300, alpha=0.9)
    labels = {node: str(index) for index, node in enumerate(selected_dfs)}
    nx.draw_networkx_labels(G, pos=layout, labels=labels, font_size=8, font_weight='bold', font_color='black')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=12, label='Source'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', markersize=12, label='Complex'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=12, label='Recent')
    ], loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def select_dataframes_by_priority(code, current_dataframes=None, max_dataframes=5, display=False):
    visitor = ASTVisitor()
    visitor.record_calls = True
    visitor.visit(ast.parse(code))
    G = nx.DiGraph()
    for edge, calls in visitor.dataframe_graph_edges.items():
        if not current_dataframes or (edge[1] in current_dataframes):
            G.add_edge(*edge, calls=set(calls))
    print(G.nodes)
    base_dfs = set([v for u, v in G.edges() if u == "[SOURCE]"])
    complex_dfs = set()
    for _, v, data in G.edges(data=True):
        if data["calls"].intersection(COMPLEX_DATAFRAME_METHODS):
            complex_dfs.add(v)
    dfs = base_dfs.union(complex_dfs)
    max_remaining_dfs = max_dataframes - len(base_dfs) - len(complex_dfs)
    if max_remaining_dfs > 0:
        reverse_bfs_order = list(nx.bfs_tree(G, next(iter(base_dfs))).nodes)[::-1]
        for df in reverse_bfs_order:
            if max_remaining_dfs <= 0:
                break
            if df not in dfs:
                dfs.add(df)
                max_remaining_dfs -= 1
    all_dataframes = list(dict.fromkeys(visitor.dataframes))
    dfs_order_index = {i: df for i, df in enumerate(all_dataframes) if df in dfs}
    ordered_dfs = list(dict(sorted(dfs_order_index.items())).values())
    if len(dfs) > max_dataframes:
        ordered_dfs = ordered_dfs[-max_dataframes:]
    if display:
        display_dataframe_graph(G, base_dfs, complex_dfs, ordered_dfs)
    return ordered_dfs

def extract_code_from_response(text):
    pattern = r'```(?:python\s+)?(.*?)(?:```|$)|<code>(.*?)</|<python>(.*?)(?:</|$)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        code_block = next(group for group in match.groups() if group)
        return code_block.strip(), text
    return text, text