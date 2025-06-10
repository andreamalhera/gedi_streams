#  Stream of Intent: Enabling Streaming Process Mining by Generating Intentional Event Streams 

Stream Of Intent is a framework for generating synthetic event streams with intentional features. It is designed to address the need for realistic and configurable event data for the evaluation of Streaming Process Mining (SPM) algorithms.

## Overview

The analysis of real-time event streams is a growing area of process mining. However, the evaluation of SPM algorithms is often hindered by the lack of realistic and configurable event stream data. Existing approaches often rely on the streamification of static event logs, which do not capture the complexities of real-world streaming environments, such as out-of-order events, concurrent activities, and concept drifts.

Stream Of Intent solves this problem by providing a novel framework for generating event streams with specific, intentional characteristics. It combines process tree generation, Markov chain modeling, and Bayesian optimization to produce event streams that are not only realistic but also tailored to specific evaluation scenarios. This allows researchers and practitioners to benchmark and develop SPM algorithms in a more targeted and reproducible manner.

## How it Works

The Stream Of Intent pipeline is a multi-phase architecture that generates event streams with intentional features. The process can be summarized as follows:

1.  **Configuration**: The process starts with static and dynamic configuration parameters. Static parameters include settings like window size, while dynamic parameters are subject to optimization. The user can also define target values for one or more stream features.
2.  **Process Tree Generation**: A process tree is generated using stochastic methods based on the initial configuration parameters. The `create_PTLG` function in `/generator/model.py` is responsible for this step.
3.  **Markov Chain Conversion**: The generated process tree is then converted into a Markov chain, which serves as the underlying model for the event simulation. This is handled by the utilities in `/def_configurations/utils/def_utils.py`.
4.  **Event Stream Simulation**: The Distributed Event Factory (DEF) uses the Markov chain to simulate a realistic event stream, taking into account concurrency, event lifecycles, and durations. The simulation logic is located in `/generator/simulation.py`.
5.  **Feature Extraction and Evaluation**: The simulated stream is processed in windows, and stream features are extracted and evaluated against the target values. Feature extraction is implemented in the `/features` directory.
6.  **Bayesian Optimization**: A Bayesian optimization procedure iteratively adjusts the generator's parameters based on the evaluation to meet the desired feature specifications.
7.  **Output**: Once the optimization is complete, Stream Of Intent produces a static definition that can be used by DEF to replay the stream with the specified features.

## Features

* **Configurable Event Stream Generation**: Stream Of Intent allows for the generation of event streams with a wide range of configurable characteristics, including:
- Concurrency 
- Temporal Disorder 
- Indefinite Case Lifespans
* **Intentional Feature Generation**: The framework can generate event streams that intentionally satisfy predefined meta-feature characteristics. This is achieved through a combination of a configurable stream parameter, stochastic process models, and hyperparameter optimization.
* **Extensible Feature Set**: Stream Of Intent includes a set of built-in stream features for controlling the generation process. It is also possible to define additional custom features tailored to specific use cases. The core feature extraction logic can be found in `/features/feature_extraction.py`, with specific feature implementations in `/features/simple_stream_stats.py` and `/features/streaming_features.py`.

## Installation

To use Stream Of Intent, you will need to have Python installed. You can then clone the repository and install the required dependencies:

```bash
git clone [https://github.com/andreamalhera/.git](https://github.com/andreamalhera/.git)
cd 
pip install -r requirements.txt
```

## Usage

The main entry point for running Stream Of Intent is the run.py script. You can execute it with a path to a configuration file:
Bash

python /run.py --config /path/to/your/config.json

## Configuration

The generation process is controlled by a configuration file that specifies the parameters for the generator, feature extractor, and plotter. The available parameters are defined in the /utils/param_keys directory.

Key configuration sections include:

- generator_params: Parameters for the event log generator, such as the configuration space for the process tree generator and the number of optimization trials. 
- feature_params: Parameters for the feature extractor, including the set of features to be used for controlling the generation process.