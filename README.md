# Agentic Hydrogel: An Agentic AI Framework for Hydrogel Design, Analysis and Conditional Generation

Agentic Hydrogel is a lightweight agentic framework for automated hydrogel material data analysis. It supports regression, classification, conditional generaiton, data analysis, visualization, metric evaluation, and report generation using a human-in-the-loop command-line workflow.

The project was developed for hydrogel material design using two dataset groups:

1. PAA hydrogel dataset (**https://doi.org/10.3390/gels10100660**)
2. Granular hydrogel dataset (**https://doi.org/10.1016/j.matt.2023.01.011**)

The project combines rule-based workflow planning, deep learning models including generative models and local/cloud summaries.

## Project Overview

The goal of this project if to establish an agentic workflow that can help a researcher move from raw material datasets to model training, conditional generation and reportable results.

The system can: 
- Load and clean hydrogel datasets
- Detect regression, classification, and generation columns
- Run adaptive PyTorch MLP models for regression/classification
- Train conditional generative models for inverse material design
- Generate material compositions from target rheological properties
- Evaluate synthetic data using distributional and conditional metrics
- Save plots, models, reports, workflow states, and generated data
- Provide LLM-generated workflow summaries

The LLM is not the scientific source of truth. It is rather used for explanation and summarization.

## Supported Tasks

1. Regression

The system can train MLP regression models. For example, in the PAA hydrogel dataset, material and printing parameters can be used to predict:

- Storage modulus
- Loss modulus

The adaptive MLP increases model complexity if performance is below a predefined threshold.

2. Classification

The system can train MLP classifiers for classification tasks when suitable target columns are present.

3. Conditional Generation

The system supports inverse design using conditional generative models.

Given target condition values such as:

- Frequency
- Oscillation strain
- Oscillation stress
- Storage modulus
- Loss modulus
- Complex modulus

the system can generate candidate material and process parameters.

Current generative models:

- Conditional VAE
- Conditional DDPM

Current model-selection rule:

```
Training dataset samples <= 2000 -> CVAE 
Training dataset samples > 2000 -> DDPM
```

## Repository Structure

```
agentic_gel/ 
├── agents/ 
│ ├── materials_planner_agent.py 
│ ├── materials_analysis_agent.py 
│ ├── materials_generation_agent.py 
│ └── materials_orchestrator_agent.py 
│
├── data/ 
│ ├── PAA_Hydrogel_Dataset/ 
│ └── Granular_Hydrogel_Dataset/ 
│ ├── documentation/ 
│ ├── system_architecture.md 
│ ├── dataset_description.md 
│ └── usage_guide.md 
│
├── notebooks/ 
│ ├── 01_paa_regression_results.ipynb 
│ ├── 02_paa_generation_results.ipynb 
│ ├── 03_granular_generation_results.ipynb 
│ └── 04_paper_figures.ipynb 
│ ├── outputs/ 
│ ├── generated_data/ 
│ ├── models/ 
│ ├── plots/ 
│ ├── reports/ 
│ └── workflows/ 
│
├── utils/ 
│ ├── adaptive_mlp.py 
│ ├── generation_cvae.py 
│ ├── generation_ddpm.py 
│ ├── generation_metrics.py 
│ ├── generation_postprocessing.py 
│ ├── hf_llm_client.py 
│ ├── ollama_llm_client.py 
│ ├── material_dataset_utils.py 
│ ├── material_preprocessing.py 
│ ├── material_report_writer.py 
│ ├── material_schema_discovery.py 
│ ├── material_visualization.py 
│ └── workflow_state.py 
│ 
├── main_materials_llm.py 
├── README.md 
├── requirements.txt 
└── .gitignore
```

## Datasets

The project expects the following dataset folders:
```
data/PAA_Hydrogel_Dataset/
data/Granular_Hydrogel_Dataset/
```
Example PAA generation conditions:
```
Frequency (Hz)
Storage modulus (Pa)
Loss modulus (Pa)
```
Example PAA generation outputs:
```
Acrylamide Conc. %
Bis-acrylamide conc %
Photo-initiator conc. %
Layer Height. (micron)
Bottom Layer exposure time (s)
Exposure time (s)
```
Example granular OscStrain generation conditions:
```
Oscillation strain (%)
Storage modulus (Pa)
Loss modulus (Pa)
Complex modulus (Pa)
```
Example granular OscStress generation conditions:
```
Oscillation stress (Pa)
Storage modulus (Pa)
Loss modulus (Pa)
Complex modulus (Pa)
```
Example granular generation outputs:
```
Microgel Conc. (%)
Microgel Size (um)
Fluid Phase Ca2+ (mM)
Fluid Phase Na+ (mM)
Est. Volume Fraction
Resuspension_encoded
```
Metadata columns such as exported row indices and Stress/Strain labels are automatically removed during dataset loading.

## Installation

Create and activate a conda environment:
```bash
conda create -n agentic_gel_env python=3.14.3
conda activate agentic_gel_env
```
Install dependencies:
```bash
pip install -r requirements.txt
```
For PyTorch, install the version appropriate for your system. For CPU-only use:
```bash
pip install torch torchvision torchaudio
```
For GPU/Colab use, install the CUDA-compatible version recommended by PyTorch.

## Running the Main CLI

Installation:

Create and activate a conda environment:
```bash
conda create -n agentic_gel_env python=3.10
conda activate agentic_gel_env
```
Install dependencies:
```bash
pip install -r requirements.txt
```
For PyTorch, install the version appropriate for your system. For CPU-only use:

pip install torch torchvision torchaudio

For GPU/Colab use, install the CUDA-compatible version recommended by PyTorch.

Running the Main CLI

The main entry point is:
```bash
python main_materials_llm.py
```
The CLI supports human-in-the-loop confirmation by default. Use --yes to skip confirmation.

## LLM Support

The system works without an LLM. The deterministic rule-based workflow is the default trusted method.

Supported optional LLM modes:
```
none
rule_based
ollama
huggingface
auto
Ollama
```
Ollama can be used for local summaries.

Example:
```bash
python main_materials_llm.py ^
  --dataset paa_hydrogel ^
  --task generation ^
  --forced_model cvae ^
  --cvae_epochs 10 ^
  --llm_provider ollama ^
  --ollama_model qwen2.5:1.5b ^
  --yes
  ```
Hugging Face

Hugging Face can be used as a cloud fallback if an appropriate token and supported model are available.

Set environment variables:
```bash
set HF_TOKEN=your_huggingface_token
set HF_MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
```
Then run:
```bash
python main_materials_llm.py ^
  --dataset paa_hydrogel ^
  --task generation ^
  --forced_model cvae ^
  --cvae_epochs 10 ^
  --llm_provider huggingface ^
  --yes
```
The LLM only summarizes workflow results. It does not decide final scientific outputs.

## Citation

```
Under review @Frontiers in Soft Matter
```
