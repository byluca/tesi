# brain-tumor-graph-segmentation

This repository contains the code, and resources associated with the paper:

**Leveraging Latent Representations in Graph Neural Networks for Enhanced and
Explainable Brain Tumor Segmentation**

Authors:

DOI:

This repository includes all necessary files and instructions to reproduce the main experiments and results presented in the paper. If you find this code useful, please consider citing our paper:


## Abstract
**Background and Objective**: The precise delineation of brain tumors is vital in patient care, from initial diagnosis through treatment strategy development to outcome prediction. Historically, this process depends on radiologists manually outlining tumor boundaries—a method that demands considerable time and suffers from inconsistencies between different observers.

**Methods**: This study presents a novel approach to brain tumor segmentation by integrating latent representations derived from 3D Convolutional Neural Networks (CNNs) into Graph Neural Networks (GNNs). While CNNs effectively extract local spatial features, they often lose critical location information for accurately segmenting complex medical images like MRI scans. Conversely, GNNs excel at preserving spatial relationships but may lack detailed local feature extraction.

**Results**: By combining these two architectures, our hybrid approach leverages their strengths, significantly improving segmentation accuracy. Additionally, explainable AI techniques are used to provide interpretable insights into the decision-making process of the GNN component, further enhancing the clinical applicability of the proposed approach.

**Conclusions**: This research underscores the potential of hybrid models in advancing automated brain tumor segmentation providing valuable decision support for clinicians in identifying brain tumors.


## Requirements
* **Global requirements**: Python >= 3.8 (tested on 3.11.4 and 3.11.7)

* **System requirements**: see [requirements.txt](/requirements.txt)

```python
pip install -r requirements.txt
```
* **Environment variables (if needed)**: see [.env.example](/.env.example)

```python
cp .env.example .env
```

## Project organization

```

├── data                          <- Datasets of MR images and graphs.
│
├── images                        <- Examples images, screenshots and graphs results.
│
├── logs                          <- Logs saved during the executions of training and testing phases.
│
├── notebooks
│   ├── 1-data-exploration.ipynb  <- Explorative data analysys.
│   ├── 2-pre-processing.ipynb   <- Preprocessing pipeline with graph construction.
│   ├── 3-training.ipynb          <- Execution of training phase and results.
│   └── 4-explainability.ipynb    <- Explainability algorithms + LLMs.
│
├── reports                       <- Saved training and testing splitting and metrics.
|
├── saved                         <- Best models serialization files.
│
├── src
│   ├── helpers                   <- Configuration files and utilities functions.
│   │   ├── config.py
│   │   ├── graph.py
│   │   └── utils.py
│   │
│   ├── models                   <- Models implementation.
│   │   ├── autoencoder3d.py
│   │   └── gnn.py
│   │
│   ├── modules                  <- Python modules.
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   ├── preprocessing.py
│   │   └── training.py
│   │
│   ├── graph_constructor.py     <- Python script to construct the graph from a MRI.
│   ├── hp_tuning.py             <- Python script to execute the hyperparameter tuning.
│   └── main.py                  <- Python script for the execution of the whole pipeline.
│
├── .env.example                 <- Environment variables.
├── .gitignore                   <- Specifications of files to be ignored by Git.
├── LICENCE                      <- Licence file.
├── README.md                    <- The top-level README for developers using this project.
├── requirements.txt             <- The project requirements to be installed.

```

## Resources (TO BE ADDED)

Models training was carried out on the following resources:

* CPU: 1 x Intel(R) Xeon(R) Gold 5317 CPU @ 3.00GHz (12 cores).
* GPU: 1 x NVIDIA A100 PCIe 80GB.
* RAM: 90 GB.



## Results (TO BE ADDED)

#### Example of segmentation models 4-channels input
![model input](/images/model_input.png)
#### Example of segmentation models labels
![model output](/images/model_output.png)
#### Example of segmentation models predictions
![model prediction](/images/model_prediction.png)
#### Segmentation models results
![results](/images/metrics.png)

#### Language models generated medical reports

1. **[English outputs](/prompts/en)**
2. **[Italian outputs](/prompts/it)**


## Authors

The present project has been realized by me **[@albertovalerio](https://github.com/albertovalerio)** and my colleague ... .

## Acknowledgments (TO BE ADAPTED)

- AutoEncoder model and metrics adapted from **[@Project-MONAI](https://monai.io/)** implementation.
- Brain atlas refers to **[@Julich-Brain-Cytoarchitectonic-Atlas](https://julich-brain-atlas.de/)** institute.
- Language models taken from **[@HuggingFace](https://huggingface.co/)** community and **[@Groq](https://groq.com/)** API.
- Dataset from **[@BraTS-2023](https://www.synapse.org/#!Synapse:syn51156910/wiki/622351)** challenge.

#### Further acknowledgments (TO BE ADAPTED)

- **[@NiBabel](https://nipy.org/nibabel/)**
- **[@Nilearn](https://nilearn.github.io/)**
- **[@SIIBrA (Software Interface for Interacting with Brain Atlases)](https://siibra-python.readthedocs.io/)**
- **[@PyTorch](https://pytorch.org/)**
- **[@SpaCy](https://spacy.io/)**
- **[@NumPy](https://numpy.org/)**
- **[@Matplotlib](https://matplotlib.org/)**

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License. See [LICENSE](/LICENSE) for more information.