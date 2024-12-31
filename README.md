## Domain-Aware Prompting with LLMs for Data Science Notebooks

### **Abstract**

LLMs have shown impressive capabilities across a variety of tasks, such as semantic understanding, rea-
soning capabilities, and code synthesis. This has the potential to assist data scientists in data preparation
and exploration tasks. Computational notebooks have become popular tools for data scientists to explore,
analyse and prepare data. Notebooks provide a richer contexual environment in comparison to single-step
code generation tasks and benchmarks due to its live execution state and multi-step nature.

We build upon the ARCADE benchmark and experiments and propose novel prompting tech-
niques that leverages contextual information in a stateful notebook environment. We explore combining
execution state with various prompting methods to improve on the original ARCADE benchmark results
and evaluate our prompting methods with the Llama 3 family of models. Our experiments highlight
the importance designing domain-specific prompting that takes into context all available information to
enable stronger model performance on data science tasks in a stateful notebook environment.

![image](https://github.com/user-attachments/assets/f3d67234-8b83-4b84-8d82-603785d926e4)


### **Project Structure**

#### **Source Code**

- **`execution.py`**  
  Generates execution metadata resources, including variables, outputs, and runtime information.

- **`analysis.py`**  
  Parses and extracts meta-information from code, such as structure, dependencies, and execution details.

- **`experiments.py`**  
  Core functionality for generating predictions, running experiments, and creating datasets.

- **`explore.py`**  
  Gradio application for exploring and visualizing experiment results interactively.

- **`llm.py`**  
  Interfaces with Large Language Models (LLMs).

- **`multistep.py`**  
  Manages multi-step message chains.

- **`prompt_templates.py`**  
  Contains prompt template strings.

- **`prompt_utils.py`**  
  Utility functions for prompt generation.

- **`prompts.py`**  
  Core functions for building and managing prompts.

### **Directory Overview**

- **Notebooks**  
  Prototyping and experimentation notebooks.

- **Models**  
  Tokenizer and code interface for LLaMA 3 models.

- **Resources**  
  Extracted execution information, return types, and exemplars for prompts.

- **Artifacts**  
  Raw datasets and notebooks forming the ARCADE dataset.

- **Datasets**  
  Generated prompt datasets for the different experiments.

- **arcade_nl2code**  
  Evaluation code and utilities for building the initial dataset from the original ARCADE paper.\
  Please refer to the [ARCADE](https://github.com/google-research/arcade-nl2code) repository for instructions on building the original dataset. \
  \
  [Natural Language to Code Generation in Interactive Data Science Notebooks](https://aclanthology.org/2023.acl-long.9) (Yin et al., ACL 2023)
