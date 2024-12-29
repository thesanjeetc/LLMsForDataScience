import os
import json
import pandas as pd
import gradio as gr

EXPERIMENTS_ROOT = "/root/FYP/experiments"
EXPERIMENT_NAME = "fyp.DFS+OUT+SCH+FS(AST)"

with open(os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_NAME, "results.json"), "r") as f:
    data = json.loads(f.read())

df = pd.DataFrame(data)
df["correct"] = df["predictions"].apply(lambda preds: any(p["accuracy"] == 1.0 for p in preds))
df['model_dataset'] = df['model'].str.split("_").str[-1] + ' - ' + df['dataset_src']
df_preds = df.explode("predictions")
df_preds["correct"] = df_preds["predictions"].apply(lambda p: p["accuracy"] == 1.0)
df_preds['model_dataset'] = df_preds['model'].str.split("_").str[-1] + ' - ' + df_preds['dataset_src']
df_preds["execution_error"] = df_preds["predictions"].apply(lambda p: p.get("execution_error", False))
df_preds["error_type"] = df_preds["predictions"].apply(lambda p: p["error_text"].split(":", 1)[0].strip() if p.get("error_text") else None)
df_preds["error_text"] = df_preds["predictions"].apply(lambda p: p.get("error_text", None))
df_preds["runtime_error_type"] = df_preds["predictions"].apply(lambda p: p["error_text"].split(":")[1].strip() if "RuntimeError" in p.get("error_text", "") else None)
df_logical_errors = df_preds[(df_preds["correct"] == False) & (df_preds["execution_error"] == False)]
df_preds = df_preds.drop_duplicates(subset=["notebook_name", "turn_index", "model_dataset", "runtime_error_type"])

models = ["LLAMA3_INSTRUCT_70B", "LLAMA3_INSTRUCT_8B"] 
dataset_sources = ["existing_tasks", "new_tasks"]
filtered_df = pd.DataFrame()

key_runtime_errors = ["AttributeError", "IndexError", "KeyError", "NameError", "TypeError", "ValueError"]
runtime_errors = df_preds[df_preds["runtime_error_type"].isin(key_runtime_errors)]["runtime_error_type"].value_counts()
runtime_errors = runtime_errors.append(pd.Series({"Parse Error": len(df_preds[df_preds["error_type"] == "Parse Error"]), "Logical Error": len(df_logical_errors)}))
errors_breakdown = (runtime_errors / runtime_errors.sum() * 100).round(2).to_frame().reset_index()
errors_breakdown.columns = ["Name", "Proportion"]

error_names = ["Logical Error", "Parse Error", "Runtime - AttributeError", "Runtime - IndexError", "Runtime - KeyError", "Runtime - NameError", "Runtime - TypeError", "Runtime - ValueError"]


def show_details(row_index):
    global filtered_df
    selected_row = filtered_df.iloc[row_index]
    intent = selected_row["intent"]["value"]
    reference = selected_row["reference"]
    reference_html = ""
    if selected_row.get("ref_output_html"):
        reference_html = "\n\n".join(selected_row.get("ref_output_html").values())
    prediction = selected_row["predictions"]["code"]
    original_output = selected_row["predictions"]["original"]
    output_html = "\n\n".join(selected_row["predictions"].get("output_html", {}).values())
    error_text = selected_row["error_text"]
    prompt = selected_row["prompt_input"]
    return intent, reference, reference_html, prediction, output_html, original_output, error_text, prompt

def get_errors(error_type, model, dataset_src, task_type):
    global filtered_df
    if error_type == "Logical Error":
        filtered_df = df_preds[(df_preds["correct"] == False) & (df_preds["execution_error"] == False)]
    elif error_type == "Parse Error":
        filtered_df = df_preds[df_preds["error_type"] == "Parse Error"]
    else:
        filtered_df = df_preds[df_preds["runtime_error_type"] == error_type.rsplit(" ", 1)[-1]]
    if model:
        filtered_df = filtered_df[filtered_df["model"] == model]
    if dataset_src:
        filtered_df = filtered_df[filtered_df["dataset_src"] == dataset_src]
    if task_type:
        filtered_df = filtered_df[filtered_df["task_type"] == task_type]
    return filtered_df

with gr.Blocks() as demo:
    error_name_dropdown = gr.Dropdown(label="Error", choices=error_names)
    model_dropdown = gr.Dropdown(label="Model", choices=models)
    dataset_src_dropdown = gr.Dropdown(label="Dataset Source", choices=dataset_sources)
    task_type_dropdown = gr.Dropdown(label="Task Type", choices=["TRANSFORM", "INFO", "VISUAL"])
    errors = gr.Dataframe(value=errors_breakdown, interactive=False)
    data_output = gr.Dataframe(interactive=False)
    intent = gr.Text(label="Intent")
    reference = gr.Code(label="Reference", language="python")
    reference_html = gr.HTML(label="Reference Output")
    prediction = gr.Code(label="Prediction", language="python")
    output_html = gr.HTML(label="Prediction Output")
    original_output = gr.Code(label="Original", language="python")
    error_text = gr.Textbox(label="Error Text")
    prompt_text = gr.Textbox(label="Prompt")


    def update_table(error_type, model, dataset_src, task_type):
        global filtered_df
        filtered_df = get_errors(error_type, model, dataset_src, task_type)
        return filtered_df

    def on_row_select(evt: gr.SelectData):
        return show_details(evt.index[0])
    
    inputs = [error_name_dropdown, model_dropdown, dataset_src_dropdown, task_type_dropdown]
    error_name_dropdown.change(fn=update_table, inputs=inputs, outputs=data_output)
    model_dropdown.change(fn=update_table, inputs=inputs, outputs=data_output)
    dataset_src_dropdown.change(fn=update_table, inputs=inputs, outputs=data_output)
    task_type_dropdown.change(fn=update_table, inputs=inputs, outputs=data_output)
    data_output.select(on_row_select, outputs=[intent, reference, reference_html, prediction, output_html, original_output, error_text, prompt_text])

if __name__ == "__main__":
    demo.launch(server_port=1234)
