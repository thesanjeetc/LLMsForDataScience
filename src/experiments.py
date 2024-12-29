import os
import json
import subprocess
from datetime import datetime
from tqdm import tqdm
from concurrent import futures
import multiprocessing
import inspect
from codebleu import calc_codebleu
from src.analysis import analyse_code
import logging
from src.execution import get_execution_state
from src.prompts import build_task_prompt
from src.prompt_utils import get_num_tokens, extract_code_from_response
from arcade_nl2code.annotated_dataset import dataset as dataset_module
from .utils import *

EXPERIMENTS_ROOT = "experiments"
DATASETS_ROOT = "datasets"
ARTIFACTS_ROOT = "artifacts"

NUM_WORKERS = 200

def generate_predictions(generate_fn, model, predictions_path, dataset_path, extract_fn=None, test=False):
    open(predictions_path, 'a').close()
    if not extract_fn:
        extract_fn = extract_code_from_response
    with open(dataset_path, "r") as f:
        dataset = json.loads(f.read())
    finished_eps = set([e["metadata"]["notebook_name"] for e in from_jsonl(predictions_path)])
    dataset = [dataset[i] for i in [78, 75, 118, 72, 11, 0, 51, 25]] if test else dataset
    if len(finished_eps) == len(dataset):
        return
    for episode in tqdm(dataset, desc=f"Notebooks ({model.lower()})"):
        if episode["notebook_name"] not in finished_eps:
            episode_prediction = dict(metadata={k: v for k, v in episode.items() if k != 'turns'}, turns=[])
            pending_tasks = []
            with futures.ThreadPoolExecutor(NUM_WORKERS) as executor:
                for turn in episode['turns']:
                    pending_tasks.append([executor.submit(*args) for args in generate_fn(model, turn)])
                for i, turn in tqdm(enumerate(episode['turns']), episode["notebook_name"]):
                    results = [f.result() for f in tqdm(futures.as_completed(pending_tasks[i]))]
                    extracted_code_strs, code_strs = zip(*map(extract_fn, results))
                    turn_pred_entry = dict(metadata=dict(example=turn, model=model), predictions=extracted_code_strs, original=code_strs)
                    episode_prediction['turns'].append(turn_pred_entry)
            append_jsonl(episode_prediction, predictions_path)
    from_jsonl(predictions_path, write_json=True)

def generate_batch(dataset_path):
    dataset = dataset_module.load_dataset(dataset_path)
    tasks = []
    for episode in dataset:
        for i, turn in enumerate(episode['turns']):
            uid = f"{episode['notebook_name']}/{i}"
            tasks.append((uid, turn.input))
    return tasks

def run_experiment(config, predictions_only=False, evaluations_only=False, test=False):
    current_dir = os.getcwd()
    experiment_base = "test_experiments" if test else EXPERIMENTS_ROOT
    experiment_dir = os.path.join(current_dir, experiment_base, config['name'])
    dataset_name = config['dataset'].get('dataset_name', config['name'])
    dataset_path = os.path.join(DATASETS_ROOT, f"dataset.{dataset_name}.json")
    results_path = os.path.join(experiment_dir, "results.json")
    config_path = os.path.join(experiment_dir, "config.json")
    artifacts_dir = os.path.join(current_dir, ARTIFACTS_ROOT)
    log_path = os.path.join(experiment_dir, "debug.log")

    if config.get("arcade", False):
        generate_arcade_dataset(**config["dataset"])
    elif config["dataset"].get("base_experiment_name"):
        generate_dervived_dataset(dataset_name, **config["dataset"])
    else:
        generate_dataset(dataset_name, config["dataset"])

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    
    if not evaluations_only:
        with open(config_path, "w") as f:
            generate_fn_str = {"generate_fn": None} # inspect.getsource(config["generate_fn"])}
            extract_fn_str = {"extract_fn": None} # inspect.getsource(config["extract_fn"])}
            datetime_str = {"start": datetime.now().strftime("%I:%M%p - %B %d, %Y")}
            config["dataset"] |= {"response_fn": None}
            f.write(json.dumps(config | generate_fn_str | extract_fn_str | datetime_str, indent=2))

        processes = []
        
        for model in config["models"]:
            if config["dataset"].get("base_experiment_name"):
                dataset_path = os.path.join(DATASETS_ROOT, f"dataset.{dataset_name}.{model.lower()}.json")
            predictions_path = os.path.join(experiment_dir, f"predictions.{model.lower()}.json")
            p = multiprocessing.Process(
                target=generate_predictions,
                args=(config["generate_fn"], model, predictions_path, dataset_path, config.get("extract_fn"), test)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    if predictions_only:
        return
    
    for model in config["models"]:
        predictions_path = os.path.join(experiment_dir, f"predictions.{model.lower()}.json")
        evaluate_experiment(experiment_dir, predictions_path, artifacts_dir)

    if os.path.exists(results_path):
        with open(results_path) as f:
            data = f.read()
            if all(m in data for m in config["models"]):
                return

    results = []

    for model in config["models"]:
        predictions_path = os.path.join(experiment_dir, f"predictions.{model.lower()}.json")
        results.extend(get_evaluation_metrics(predictions_path, model))

    with open(results_path, "w") as f:
        f.write(json.dumps(results, indent=2, default=list))

def evaluate_experiment(experiment_dir, predictions_path, artifacts_dir):
    docker_image = "superninja007/notebook_evaluator:latest"

    num_workers = 6
    num_local_workers = 2

    command = [
        "docker", "run", "--shm-size=4g",
        "--mount", f"type=bind,source={experiment_dir},target=/data",
        "--mount", f"type=bind,source={artifacts_dir},target=/artifacts",
        "-w", "/",
        "--entrypoint", "/opt/conda/bin/python",
        docker_image,
        "-m", "arcade_nl2code.evaluation.execution_evaluation_main",
        "--prediction_file", f"/data/{predictions_path.split('/')[-1]}",
        "--output_path", "/data/",
        "--runtime_artifact_root", "/artifacts",
        "--lm_output_postprocessor", "extract_first_cell_block",
        "--split_episode",
        "--noreuse_state",
        "--self_consistency_decoding",
        "--timeout", "180",
        "--num_workers", str(num_local_workers),
        "--num_work_units", str(num_workers),
    ]
    
    if "\"eval_results\"" not in open(predictions_path).read():
        processes = []

        for work_unit_id in range(num_workers):
            process = subprocess.Popen([
                *command, 
                "--work_unit_id", str(work_unit_id)
            ])
            processes.append(process)
        
        for process in processes:
            process.wait()


def get_evaluation_metrics(predictions_path, model):
    with open(predictions_path, "r") as f:
        predictions = json.loads(f.read())

    tasks = []

    for notebook in predictions:
        for i, turn in enumerate(notebook["turns"]):
            task = {}
            preds = turn["predictions"]
            ref = turn["metadata"]["example"]["turn"]["code"]["value"]

            task["notebook_name"] = notebook["metadata"]["notebook_name"]
            task["turn_index"] = i
            task["dataset_src"] = notebook["metadata"]["dataset_src"]
            task["model"] = model

            if turn["metadata"]["example"]["metadata"].get("notebook_context_prompt", None):
                # task["input_msgs"] = turn["metadata"]["example"]["input_msgs"]
                task["num_context_cells"] = turn["metadata"]["example"]["metadata"]["num_context_cells"]
                task["total_context_cells"] = turn["metadata"]["example"]["metadata"]["total_context_cells"]
                task["prompt_length"] = turn["metadata"]["example"]["metadata"]["prompt_size"]
            else:
                task["prompt_length"] = turn["metadata"]["example"]["metadata"]["prompt_length"]
                task["num_context_cells"] = len(turn["metadata"]["example"]["metadata"]["context_cells"])
                task["num_code_context_cells"] = len([c for c in turn["metadata"]["example"]["metadata"]["context_cells"] if c["cell_type"] == "code"])

            task["reference"] = turn["metadata"]["example"]["turn"]["code"]["value"]
            task["code_context"] = turn["metadata"]["example"]["turn"]["code_context"]
            task["intent"] = turn["metadata"]["example"]["turn"]["intent"]
            task["prompt_input"] = turn["metadata"]["example"]["input"]
            task["clusters"] = turn.get("hyp_clusters")
            task["ref_output_html"] = turn.get("ref_output_html")

            if len(preds) != len(turn["eval_results"]):
                turn["eval_results"] = turn["eval_results"] * len(preds)

            task["predictions"] = [
                {"code": p}
                | {"original": turn["original"][i]}
                | turn["eval_results"][i]
                | calc_codebleu([ref], [p], lang="python")
                | {"analysis": analyse_code(task["code_context"], p)}
                # | {"output_html": None}
                | {"correct": turn["eval_results"][i].get("accuracy", 0.0) == 1.0}
                for i, p in enumerate(preds)
            ]

            task["reference_analysis"] = analyse_code(task["code_context"], task["reference"])
            tasks.append(task)

    return tasks

def generate_arcade_dataset(
    dataset_name,
    add_exemplars=False,
    max_prompt_size = 900,
    max_notebook_context_length = 1200,
    prompt_style="vanilla"
):
    dataset_path =  os.path.join(DATASETS_ROOT, f"dataset.{dataset_name}.json")
    if os.path.exists(dataset_path):
        return

    schema_repr_method = "originating_dfs.header_description.after_variable_cell"

    prompt_styles = {
        "vanilla": "short_code_no_preamble", 
        "step_by_step": "step_only_no_preamble", 
        "step_by_step+preamble": "step_only", 
        "step_by_step+preamble+explanation": "step+explanation"
    }

    cmd = [
        "faketime", "2022-12-10 12:00:00", 
        "python", "-m",
        "arcade_nl2code.annotated_dataset.generate_schema_augmented_prompts",
        "--schema_representation_method", schema_repr_method,
        "--max_prompt_size", str(max_prompt_size),
    ]

    if add_exemplars:
        cmd.extend([
            "--max_notebook_context_len", str(max_notebook_context_length),
            "--add_exemplars",
            "--exemplar_notebook", "arcade_nl2code/annotated_dataset/resources/prompt_exemplar_templates.ipynb",
            "--format_configs", prompt_styles[prompt_style],
            "--exemplar_index", "1,2,3,5"
        ])
        
    base_dataset_path = os.path.join(DATASETS_ROOT, "dataset.json")
    dataset_path = f"dataset.{dataset_name}.json"

    subprocess.run([
        *cmd,
        "--dataset", base_dataset_path,
        "--output_folder", DATASETS_ROOT,
        "--output_dataset_name", dataset_path,
        "--runtime_artifacts_root", ARTIFACTS_ROOT
    ])

def generate_dataset(dataset_name, notebook_prompt_config):
    dataset_path =  os.path.join(DATASETS_ROOT, f"dataset.{dataset_name}.json")
    if os.path.exists(dataset_path):
        return
    base_dataset_path = os.path.join("datasets", "dataset.formatted.json")
    with open(base_dataset_path, "r") as f:
        dataset = json.load(f)
    for notebook in dataset:
        execution_info = get_execution_state(notebook["notebook_name"])
        for i, task in enumerate(notebook["turns"]):
            cell_execution_info = execution_info[i]
            prompt_messages, notebook_prompts = build_task_prompt(task, cell_execution_info, notebook_prompt_config)
            prompt_str = "\n".join(list(map(lambda m: m["content"], prompt_messages)))
            task["input"] = prompt_str
            task["input_msgs"] = prompt_messages
            task["metadata"] |= notebook_prompts
            task["metadata"]["prompt_size"] = get_num_tokens(prompt_str)
    with open(dataset_path, "w") as f:
        f.write(json.dumps(dataset, indent=2))


def generate_dervived_dataset(experiment_name, base_experiment_name, response_fn):
    dataset = json.loads(open(os.path.join("datasets", f"dataset.{base_experiment_name}.json")).read())
    config = json.loads(open(os.path.join("experiments", base_experiment_name, f"config.json")).read())
    for m in config["models"]:
        m = m.lower()
        evaluation = json.loads(open(os.path.join("experiments", base_experiment_name, f"predictions.{m}.json")).read())
        for ni, n in enumerate(dataset):
            for ti, t in enumerate(n["turns"]):
                outputs = evaluation[ni]["turns"][ti]["original"]
                eval_results = evaluation[ni]["turns"][ti]["eval_results"]
                t["metadata"]["initial_eval_results"] = eval_results
                t["user_messages"] = [response_fn(o, e) for o, e in zip(outputs, eval_results)]
        dataset_path = os.path.join("datasets", f"dataset.{experiment_name}.{m}.json")
        with open(dataset_path, "w") as f:
            f.write(json.dumps(dataset, indent=2))