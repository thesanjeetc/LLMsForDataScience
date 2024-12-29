from src.llm import generate_response
from src.experiments import run_experiment
from src.prompt_templates import *
from src.prompt_utils import extract_code_from_response
from src.multistep import generate_analysis_prompt, generate_analysis_err_prompt, generate_analysis_err_out_prompt
from concurrent import futures
import argparse
from functools import partial

TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FIXED_TEMP = 0.6
MAX_TOKENS = 512
NUM_PASSES = 5

def generate_fn(model, task, num_passes=NUM_PASSES):
    return [(generate_response, model, task["input_msgs"], FIXED_TEMP, MAX_TOKENS) for i in range(num_passes)]

def generate_fn_response(model, task):
    return [(generate_response, model, task["input_msgs"] + task["user_messages"][i], FIXED_TEMP, MAX_TOKENS) for i in range(NUM_PASSES)]

def generate_fn_temperature(model, task):
    return [(generate_response, model, task["input_msgs"], t, MAX_TOKENS) for t in TEMPERATURES for i in range(NUM_PASSES)]

base_config = {
    "generate_fn": generate_fn,
    "extract_fn": extract_code_from_response,
    "models": ["LLAMA3_INSTRUCT_8B", "LLAMA3_INSTRUCT_70B"],
    "metadata":{
        "temperatures": 0.6,
        "num_passes": 5,
    }
}

baseline_experiments = {
    "arcade.vanilla-FS": base_config | { 
        "name":"arcade.vanilla-FS",
        "arcade": True,
        "dataset": { "dataset_name": "arcade.vanilla-FS"},
    },
    "arcade.CoT-FS": base_config | { 
        "name":"arcade.CoT-FS",
        "arcade": True,
        "dataset": { "dataset_name": "arcade.CoT-FS"},
    },
    "arcade.CoT-FS+EXP": base_config | { 
        "name":"arcade.CoT-FS+EXP",
        "arcade": True,
        "dataset": {"dataset_name": "arcade.CoT-FS+EXP"},
    },
}

dataset_config = dict(
    add_markdown=True,
    split_into_cells=False, 
    add_outputs=False, 
    add_new_dataframes=False, 
    append_dataframes=False,
    append_schema_info=False,
    extract_user_functions=False,
    add_exemplars = None,
    max_dataframes = 10,
    prioritize_dataframes = True,
    max_prompt_size=7000,
    max_context_cells=15
)

fyp_experiments = {
    "fyp.vanilla": base_config | {
        "name": "fyp.vanilla",
        "dataset": dataset_config | dict(
            prompt_template=VANILLA_PROMPT
        )
    },

    "fyp.DFS": base_config | {
        "name": "fyp.DFS",
        "dataset": dataset_config | dict(
            add_new_dataframes=True,
            prompt_template=eDFS_PROMPT
        )
    },

    "fyp.DFS+OUT": base_config | {
        "name": "fyp.DFS+OUT",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            prompt_template=eDFS_PROMPT
        )
    },

    "fyp.DFS+OUT+FS": base_config | {
        "name": "fyp.DFS+OUT+FS",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            add_exemplars="arcade",
            prompt_template=eDFS_PROMPT_FS
        )
    },

    "fyp.DFS+OUT+FS+SCH": base_config | {
        "name": "fyp.DFS+OUT+FS+SCH",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            add_exemplars="arcade",
            append_schema_info= True,
            prompt_template=eDFS_PROMPT_FS_SCH
        )
    },
    
    "fyp.SPLIT-DFS+OUT+FS+SCH": base_config | {
        "name": "fyp.SPLIT-DFS+OUT+FS+SCH",
        "dataset": dataset_config | dict(
            split_into_cells=True, 
            add_outputs=True,
            add_new_dataframes=True,
            add_exemplars="arcade",
            prompt_template=eDFS_PROMPT_FS_SPLIT
        )
    },

    "fyp.CoT-DFS+OUT+SCH": base_config | {
        "name": "fyp.CoT-DFS+OUT+SCH",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            append_schema_info= True,
            prompt_template=COT_eDFS_PROMPT_SCH
        )
    },

    "fyp.CoT-FS+DFS+OUT+SCH": base_config | {
        "name": "fyp.CoT-FS+DFS+OUT+SCH",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            append_schema_info= True,
            prompt_template=COT_FS_eDFS_PROMPT_SCH,
            add_exemplars = "fyp_cot"
        )
    },

    "fyp.CoT+2S-DFS+OUT+SCH": base_config | {
        "name": "fyp.CoT+2S-DFS+OUT+SCH",
        "generate_fn": generate_fn_response,
        "dataset": dict(
            base_experiment_name="fyp.CoT-DFS+OUT+SCH",
            response_fn=generate_analysis_prompt
        )
    },

    "fyp.CoT+2S+ERR-DFS+OUT+SCH": base_config | {
        "name": "fyp.CoT+2S+ERR-DFS+OUT+SCH",
        "generate_fn": generate_fn_response,
        "dataset": dict(
            base_experiment_name="fyp.CoT-DFS+OUT+SCH",
            response_fn=generate_analysis_err_prompt
        )
    },

    "fyp.CoT+2S+ERR+RES-DFS+OUT+SCH": base_config | {
        "name": "fyp.CoT+2S+ERR+RES-DFS+OUT+SCH",
        "generate_fn": generate_fn_response,
        "dataset": dict(
            base_experiment_name="fyp.CoT-DFS+OUT+SCH",
            response_fn=generate_analysis_err_out_prompt
        )
    },

    "fyp.CoT-DFS+OUT+SCH-[pass@30]": base_config | {
        "name": "fyp.CoT-DFS+OUT+SCH-[pass@30]",
        "generate_fn": partial(generate_fn, num_passes=30),
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            append_schema_info= True,
            prompt_template=COT_eDFS_PROMPT_SCH
        )
    },

    "fyp.CoT-DFS+OUT+SCH-[T]": base_config | {
        "name": "fyp.CoT-DFS+OUT+SCH-[T]",
        "generate_fn": generate_fn_temperature,
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            append_schema_info= True,
            prompt_template=COT_eDFS_PROMPT_SCH
        )
    },
}

additional_experiments = {
    "fyp.CoT-DFS+OUT": base_config | {
        "name": "fyp.CoT-DFS+OUT",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            append_schema_info= False,
            prompt_template=COT_eDFS_PROMPT
        )
    },

    "fyp.CoT-aDFS+OUT": base_config | {
        "name": "fyp.CoT-aDFS+OUT",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=False,
            append_dataframes=True,
            append_schema_info= False,
            prompt_template=COT_aDFS_PROMPT
        )
    },

    "fyp.CoT-SCH": base_config | {
        "name": "fyp.CoT-SCH",

        "dataset": dataset_config | dict(
            add_outputs=False,
            add_new_dataframes=False,
            append_schema_info= True,
            prompt_template=COT_eDFS_PROMPT_SCH
        )
    },

    "fyp.DFS+OUT+SCH+FS(AST)": base_config | {
        "name": "fyp.DFS+OUT+SCH+FS(AST)",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            add_exemplars="ast",
            append_schema_info= True,
            prompt_template=eDFS_PROMPT_SCH
        )
    },

    "fyp.DFS+OUT+FS(AST)": base_config | {
        "name": "fyp.DFS+OUT+FS(AST)",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            add_exemplars="ast",
            append_schema_info= False,
            prompt_template=eDFS_PROMPT
        )
    },

    "fyp.DFS+FS": base_config | {
        "name": "fyp.DFS+FS",
        "dataset": dataset_config | dict(
            add_outputs=False,
            add_new_dataframes=True,
            add_exemplars="arcade",
            prompt_template=eDFS_PROMPT_FS
        )
    },

    "fyp.DFS+OUT+SCH": base_config | {
        "name": "fyp.DFS+OUT+SCH",
        "dataset": dataset_config | dict(
            add_outputs=True,
            add_new_dataframes=True,
            append_schema_info= True,
            prompt_template=eDFS_PROMPT_SCH
        )
    },

    "arcade.CoT-FS-[pass@30]": base_config | { 
        "name":"arcade.CoT-FS-[pass@30]",
        "generate_fn": partial(generate_fn, num_passes=30),
        "arcade": True,
        "dataset": { "dataset_name": "arcade.CoT-FS"},
    },
}

for j in additional_experiments.values():
    run_experiment(j)