# R-Search

This project is trained. The Large Language Model (LLM) should output in the following format:

<think>Reasoning process</think> + <search>Search planning results - DAG</search> + <result>[Search results]</result> + <answer>Final generated result</answer>

## Installation

1.  **PyTorch:** Install PyTorch version 2.6.
2.  **trl:** Install from GitHub:
    ```bash
    pip install git+https://github.com/huggingface/trl.git
    ```
3.  **lagent:**
    ```bash
    cd lagent
    pip install -e .
    ```

## Data Processing

Data clustering uses `data/clusters.py`, and dataset generation and filtering use `data/qa_gen.py`.

## Training

To train the model, run the following script:

```bash
sh train.sh
``` 