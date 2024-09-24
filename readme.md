# DSI

This work is based on the paper [*Transforming Slot Schema Induction with Generative Dialogue State Inference*](https://aclanthology.org/2024.sigdial-1.27/), which trains a S2S model to discover new slots from unlabeled dialogue data. The S2S model is trained on the [D0T dataset](https://github.com/jdfinch/dot).

## Install

Create a conda environment then pip install requirements.txt

```bash
conda create --solver=libmamba -n dsi -c rapidsai -c conda-forge -c nvidia cudf=24.02 cuml=24.02 python=3.10 cuda-version=11.8
conda activate dsi
pip install -r requirements.txt
```

## DSI T5 model

The training/experiment code relies on loading a pickle object representing the D0T data. Therefore, a simpler usage of the DSI model is exemplified in `dsi/s2s_dsi.py`. The folder `s2s_dsi/` contains the training and experiment code for reference.

## Full SSI

After getting labels from the DSI s2s model (see `data/silver_...`), SBERT encoding and clustering are used to induce the final schema. Experiment code for this is in `dsi`, with `dsi/experiment.py` being the main script.
