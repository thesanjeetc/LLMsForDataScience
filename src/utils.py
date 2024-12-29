import os
import json
import pandas as pd
from arcade_nl2code.evaluation.utils import DataclassJsonEncoder

def to_jsonl(file_path):
    input_path = f"{file_path.rsplit('.', 1)[-2]}.json"
    df = pd.read_json(input_path)
    output_path = f"{file_path.rsplit('.', 1)[-2]}.log"
    df.to_json(output_path, orient='records', lines=True)

def from_jsonl(file_path, write_json = False, delete_log = False):
    input_path = f"{file_path.rsplit('.', 1)[-2]}.log"
    open(input_path, 'a').close()
    df = pd.read_json(input_path, lines=True)
    if write_json:
        output_path = f"{file_path.rsplit('.', 1)[-2]}.json"
        df.to_json(output_path, orient='records', indent=2)
        if delete_log:
            os.remove(input_path)
    return df.to_dict(orient='records')

def append_jsonl(entry, file_path):
    mode = 'a' if os.path.exists(file_path) else 'w'
    output_path = f"{file_path.rsplit('.', 1)[-2]}.log"
    with open(output_path, mode) as f:
        f.write('\n' + json.dumps(entry, cls=DataclassJsonEncoder))

def get_baseline_results(models=None, metric="pass5"):
    data = json.loads(open(os.path.join("resources", "baseline.json")).read())
    df_baseline = pd.DataFrame(data)
    df_baseline['model_dataset'] = df_baseline['model'].str.split("_").str[-1] + ' - ' + df_baseline['dataset_src']
    if models:
        df_baseline = df_baseline[df_baseline["model"].isin(models)]
    
    return df_baseline[["model", "model_dataset", "experiment_name", "dataset_src", metric]]

import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_callable_methods(module):
  return [
      m for m in dir(module) 
      if callable(getattr(module, m)) and not m.startswith("_")
  ]
  
def scrape_pandas_dataframe_return_methods():
  base_url = "https://pandas.pydata.org/docs/reference/api/"

  res = []

  for module in [pd, pd.DataFrame, pd.Series]:
    methods = get_callable_methods(module)
    for m in methods:
      base_name = module.__name__
      if module.__name__ != "pandas":
        base_name = f"pandas.{base_name}"
      url = f"{base_url}{base_name}.{m}.html"
      response = requests.get(url)
      soup = BeautifulSoup(response.text, 'html.parser')
      try:
          ret = soup.find_all(string='Returns')[0].parent.find_next_sibling("dd").dt.text
          if "Series" in ret or "DataFrame" in ret:
              res.append({
                  "base": base_name,
                  "method": m
              })
              print(m, ret)
      except:
          pass
