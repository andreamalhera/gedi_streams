<p>
  <img src="gedi_streams/utils/logo.png" alt="Logo" width="100" align="left" />
  <h1 style="display: inline;">GEDI Streams</h1>
</p>

**G**enerating **E**vent **D**ata with **I**ntentional Features for Process Mining **Streams** <br />
This repository contains the codebase based on the [GEDI paper](https://mcml.ai/publications/gedi.pdf) accepted at the <> conference.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [General Usage](#general-usage)
- [Experiments](#experiments)
- [Citation](#citation)

## Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Graphviz on your OS e.g.
For MacOS:
```console
brew install graphviz
brew install swig
```
- For smac:
```console
conda install pyrfr swig
```
## Installation
To directly use GEDI Streams methods via `import gedi_streams`, install directly from [PyPi](https://pypi.org/project/gedi/).
```console
pip install gedi_streams
```
Alternatively, you can create an environment with
- `conda env create -f .conda.yml`

Run:
```console
python -c "from gedi_streams import gedi_streams; gedi_streams('config_files/pipeline_steps/generation.json')"
```
or

```console
conda activate gedi_streams
python main.py -a config_files/test/experiment_test.json
```
The last step should take only a few minutes to run.

## General Usage
Our pipeline offers several pipeline steps, which can be run sequentially or partially ordered:
- [Feature Extraction](#feature-extraction)
- [Generation](#generation)
- [Evaluation Plotter](https://github.com/lmu-dbs/gedi_streams/blob/16-documentation-update-readme/README.md#evaluation-plotting)

To run different steps of the GEDI Streams  pipeline, please adapt the `.json` accordingly.
```console
conda activate gedi_streams
python main.py -a config_files/pipeline_steps/<pipeline-step>.json
```
For reference of possible keys and values for each step, please see `config_files/test/experiment_test.json`.
To run the whole pipeline please create a new `.json` file, specifying all steps you want to run and specify desired keys and values for each step.
To reproduce results from our paper, please refer to [Experiments](#experiments).

### Feature Extraction
---
To extract the features on the event-log level and use them for hyperparameter optimization, we employ the following script:
```console
conda activate gedi_streams
python main.py -a config_files/pipeline_steps/feature_extraction.json
```
The JSON file consists of the following key-value pairs:

- pipeline_step: denotes the current step in the pipeline (here: feature_extraction)
- input_path: folder to the input files
- feature params: defines a dictionary, where the inner dictionary consists of a key-value pair 'feature_set' with a list of features being extracted from the references files. A list of valid features can be looked up from the FEEED extractor
- output_path: defines the path, where plots are saved to
- real_eventlog_path: defines the file with the features extracted from the real event logs
- plot_type: defines the style of the output plotting (possible values: violinplot, boxplot)
- font_size: label font size of the output plot
- boxplot_width: width of the violinplot/boxplot

### Generation
---
After having extracted meta features from the files, the next step is to generate event log data accordingly. Generally, there are two settings on how the targets are defined: i) meta feature targets are defined by the meta features from the real event log data; ii) a configuration space is defined which resembles the feasible meta features space. 

The command to execute the generation step is given by a exemplarily generation.json file:

```console
conda activate gedi_streams
python main.py -a config_files/pipeline_steps/generation.json
```

In the `generation.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: event_logs_generation)
* output_path: defines the output folder
* generator_params: defines the configuration of the generator itself. For the generator itself, we can set values for the general 'experiment', 'config_space', 'n_trials', and a specific 'plot_reference_feature' being used for plotting

    - experiment: defines the path to the input file which contains the features that are used for the optimization step. The 'objectives' define the specific features, which are the optimization criteria.
    - config_space: here, we define the configuration of the generator module (here: process tree generator). The process tree generator can process input information which defines characteristics for the generated data (a more thorough overview of the params can be found [here](https://github.com/tjouck/PTandLogGenerator):

        - mode: most frequent number of visible activities
        - sequence: the probability of adding a sequence operator to the tree
        - choice: the probability of adding a choice operator to the tree
        - parallel: the probability of adding a parallel operator to the tree
        - loop: the probability of adding a loop operator to the tree
        - silent: probability to add silent activity to a choice or loop operator
        - lt_dependency: the probability of adding a random dependency to the tree
        - num_traces: the number of traces in the event log
        - duplicate: the probability of duplicating an activity label
        - or: probability to add an or operator to the tree

    - n_trials: the maximum number of trials for the hyperparameter optimization to find a feasible solution to the specific configuration being used as the target

    - plot_reference_feature: defines the feature, which is used on the x-axis on the output plots, i.e., each feature defined in the 'objectives' of the 'experiment' is plotted against the reference feature being defined in this value

In case of manually defining the targets for the features in config space, the following table shows the range of the features in the real-world event log data (BPIC's) for reference:
<div style="overflow-x:auto;">
    <table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th>n_traces</th>
        <th>n_unique_traces</th>
        <th>ratio_variants_per_number_of_traces</th>
        <th>trace_len_min</th>
        <th>trace_len_max</th>
        <th>trace_len_mean</th>
        <th>trace_len_median</th>
        <th>trace_len_mode</th>
        <th>trace_len_std</th>
        <th>trace_len_variance</th>
        <th>trace_len_q1</th>
        <th>trace_len_q3</th>
        <th>trace_len_iqr</th>
        <th>trace_len_geometric_mean</th>
        <th>trace_len_geometric_std</th>
        <th>trace_len_harmonic_mean</th>
        <th>trace_len_skewness</th>
        <th>trace_len_kurtosis</th>
        <th>trace_len_coefficient_variation</th>
        <th>trace_len_entropy</th>
        <th>trace_len_hist1</th>
        <th>trace_len_hist2</th>
        <th>trace_len_hist3</th>
        <th>trace_len_hist4</th>
        <th>trace_len_hist5</th>
        <th>trace_len_hist6</th>
        <th>trace_len_hist7</th>
        <th>trace_len_hist8</th>
        <th>trace_len_hist9</th>
        <th>trace_len_hist10</th>
        <th>trace_len_skewness_hist</th>
        <th>trace_len_kurtosis_hist</th>
        <th>ratio_most_common_variant</th>
        <th>ratio_top_1_variants</th>
        <th>ratio_top_5_variants</th>
        <th>ratio_top_10_variants</th>
        <th>ratio_top_20_variants</th>
        <th>ratio_top_50_variants</th>
        <th>ratio_top_75_variants</th>
        <th>mean_variant_occurrence</th>
        <th>std_variant_occurrence</th>
        <th>skewness_variant_occurrence</th>
        <th>kurtosis_variant_occurrence</th>
        <th>n_unique_activities</th>
        <th>activities_min</th>
        <th>activities_max</th>
        <th>activities_mean</th>
        <th>activities_median</th>
        <th>activities_std</th>
        <th>activities_variance</th>
        <th>activities_q1</th>
        <th>activities_q3</th>
        <th>activities_iqr</th>
        <th>activities_skewness</th>
        <th>activities_kurtosis</th>
        <th>n_unique_start_activities</th>
        <th>start_activities_min</th>
        <th>start_activities_max</th>
        <th>start_activities_mean</th>
        <th>start_activities_median</th>
        <th>start_activities_std</th>
        <th>start_activities_variance</th>
        <th>start_activities_q1</th>
        <th>start_activities_q3</th>
        <th>start_activities_iqr</th>
        <th>start_activities_skewness</th>
        <th>start_activities_kurtosis</th>
        <th>n_unique_end_activities</th>
        <th>end_activities_min</th>
        <th>end_activities_max</th>
        <th>end_activities_mean</th>
        <th>end_activities_median</th>
        <th>end_activities_std</th>
        <th>end_activities_variance</th>
        <th>end_activities_q1</th>
        <th>end_activities_q3</th>
        <th>end_activities_iqr</th>
        <th>end_activities_skewness</th>
        <th>end_activities_kurtosis</th>
        <th>eventropy_trace</th>
        <th>eventropy_prefix</th>
        <th>eventropy_global_block</th>
        <th>eventropy_lempel_ziv</th>
        <th>eventropy_k_block_diff_1</th>
        <th>eventropy_k_block_diff_3</th>
        <th>eventropy_k_block_diff_5</th>
        <th>eventropy_k_block_ratio_1</th>
        <th>eventropy_k_block_ratio_3</th>
        <th>eventropy_k_block_ratio_5</th>
        <th>eventropy_knn_3</th>
        <th>eventropy_knn_5</th>
        <th>eventropy_knn_7</th>
        <th>epa_variant_entropy</th>
        <th>epa_normalized_variant_entropy</th>
        <th>epa_sequence_entropy</th>
        <th>epa_normalized_sequence_entropy</th>
        <th>epa_sequence_entropy_linear_forgetting</th>
        <th>epa_normalized_sequence_entropy_linear_forgetting</th>
        <th>epa_sequence_entropy_exponential_forgetting</th>
        <th>epa_normalized_sequence_entropy_exponential_forgetting</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>[ min, max ]</td>
        <td>[ 226.0, 251734.0 ]</td>
        <td>[ 6.0, 28457.0 ]</td>
        <td>[ 0.0, 1.0 ]</td>
        <td>[ 1.0, 24.0 ]</td>
        <td>[ 1.0, 2973.0 ]</td>
        <td>[ 1.0, 131.49 ]</td>
        <td>[ 1.0, 55.0 ]</td>
        <td>[ 1.0, 61.0 ]</td>
        <td>[ 0.0, 202.53 ]</td>
        <td>[ 0.0, 41017.89 ]</td>
        <td>[ 1.0, 44.0 ]</td>
        <td>[ 1.0, 169.0 ]</td>
        <td>[ 0.0, 161.0 ]</td>
        <td>[ 1.0, 53.78 ]</td>
        <td>[ 1.0, 5.65 ]</td>
        <td>[ 1.0, 51.65 ]</td>
        <td>[ -0.58, 111.97 ]</td>
        <td>[ -0.97, 14006.75 ]</td>
        <td>[ 0.0, 4.74 ]</td>
        <td>[ 5.33, 12.04 ]</td>
        <td>[ 0.0, 1.99 ]</td>
        <td>[ 0.0, 0.42 ]</td>
        <td>[ 0.0, 0.4 ]</td>
        <td>[ 0.0, 0.19 ]</td>
        <td>[ 0.0, 0.14 ]</td>
        <td>[ 0.0, 10.0 ]</td>
        <td>[ 0.0, 0.02 ]</td>
        <td>[ 0.0, 0.04 ]</td>
        <td>[ 0.0, 0.0 ]</td>
        <td>[ 0.0, 2.7 ]</td>
        <td>[ -0.58, 111.97 ]</td>
        <td>[ -0.97, 14006.75 ]</td>
        <td>[ 0.0, 0.79 ]</td>
        <td>[ 0.0, 0.87 ]</td>
        <td>[ 0.0, 0.98 ]</td>
        <td>[ 0.0, 0.99 ]</td>
        <td>[ 0.2, 1.0 ]</td>
        <td>[ 0.5, 1.0 ]</td>
        <td>[ 0.75, 1.0 ]</td>
        <td>[ 1.0, 24500.67 ]</td>
        <td>[ 0.04, 42344.04 ]</td>
        <td>[ 1.54, 64.77 ]</td>
        <td>[ 0.66, 5083.46 ]</td>
        <td>[ 1.0, 1152.0 ]</td>
        <td>[ 1.0, 66058.0 ]</td>
        <td>[ 34.0, 466141.0 ]</td>
        <td>[ 4.13, 66058.0 ]</td>
        <td>[ 2.0, 66058.0 ]</td>
        <td>[ 0.0, 120522.25 ]</td>
        <td>[ 0.0, 14525612122.34 ]</td>
        <td>[ 1.0, 66058.0 ]</td>
        <td>[ 4.0, 79860.0 ]</td>
        <td>[ 0.0, 77290.0 ]</td>
        <td>[ -0.06, 15.21 ]</td>
        <td>[ -1.5, 315.84 ]</td>
        <td>[ 1.0, 809.0 ]</td>
        <td>[ 1.0, 150370.0 ]</td>
        <td>[ 27.0, 199867.0 ]</td>
        <td>[ 3.7, 150370.0 ]</td>
        <td>[ 1.0, 150370.0 ]</td>
        <td>[ 0.0, 65387.49 ]</td>
        <td>[ 0.0, 4275524278.19 ]</td>
        <td>[ 1.0, 150370.0 ]</td>
        <td>[ 4.0, 150370.0 ]</td>
        <td>[ 0.0, 23387.25 ]</td>
        <td>[ 0.0, 9.3 ]</td>
        <td>[ -2.0, 101.82 ]</td>
        <td>[ 1.0, 757.0 ]</td>
        <td>[ 1.0, 16653.0 ]</td>
        <td>[ 28.0, 181328.0 ]</td>
        <td>[ 3.53, 24500.67 ]</td>
        <td>[ 1.0, 16653.0 ]</td>
        <td>[ 0.0, 42344.04 ]</td>
        <td>[ 0.0, 1793017566.89 ]</td>
        <td>[ 1.0, 16653.0 ]</td>
        <td>[ 3.0, 39876.0 ]</td>
        <td>[ 0.0, 39766.0 ]</td>
        <td>[ -0.7, 13.82 ]</td>
        <td>[ -2.0, 255.39 ]</td>
        <td>[ 0.0, 13.36 ]</td>
        <td>[ 0.0, 16.77 ]</td>
        <td>[ 0.0, 24.71 ]</td>
        <td>[ 0.0, 685.0 ]</td>
        <td>[ -328.0, 962.0 ]</td>
        <td>[ 0.0, 871.0 ]</td>
        <td>[ 0.0, 881.0 ]</td>
        <td>[ 0.0, 935.0 ]</td>
        <td>[ 0.0, 7.11 ]</td>
        <td>[ 0.0, 7.11 ]</td>
        <td>[ 0.0, 8.93 ]</td>
        <td>[ 0.0, 648.0 ]</td>
        <td>[ 0.0, 618.0 ]</td>
        <td>[ 0.0, 11563842.15 ]</td>
        <td>[ 0.0, 0.9 ]</td>
        <td>[ 0.0, 21146257.12 ]</td>
        <td>[ 0.0, 0.76 ]</td>
        <td>[ 0.0, 14140225.9 ]</td>
        <td>[ 0.0, 0.42 ]</td>
        <td>[ 0.0, 15576076.83 ]</td>
        <td>[ 0.0, 0.51 ]</td>
        </tr>
    </tbody>
    </table>
</div>

### Evaluation Plotting
The purpose of the evaluation plotting step is used just for visualization. Some examples of how the plotter can be used is shown in the following exemplarily script:

```console
conda activate gedi_streams
python main.py -a config_files/pipeline_steps/evaluation_plotter.json
```

Generally, in the `evaluation_plotter.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: evaluation_plotter)
* input_path: defines the input file or the input folder which is considered for the visualizations. If a single file is specified, only the features in that file are considered whereas in the case of specifying a folder, the framework iterates over all files and uses them for plotting
* plot_reference_feature: defines the feature that is used on the x-axis on the output plots, i.e., each feature defined in the input file is plotted against the reference feature being defined in this value
* targets: defines the target values which are also used as reference. Likewise to the input_path, the targets can be specified by a single file or by a folder
* output_path: defines where to store the plots

## Experiments
In this repository, experiments can be run selectively or from scratch, as preferred. For this purpose, we linked both inputs and outputs for each stage. In this section, we present the reproduction of generated event data, as in our paper, as well as the [visualization of evaluation figures](#visualizations).
We present two settings for generating intentional event logs, using [real targets](#generating-data-with-real-targets) or using [grid targets](#generating-data-with-grid-targets). Both settings output `.xes` event logs, `.json` and `.csv` files containing feature values, as well as evaluation results, from running a [process discovery benchmark](#benchmark), for the generated event logs.

### Generating data with real targets
To execute the experiments with real targets, we employ the [experiment_real_targets.json](config_files/experiment_real_targets.json). The script's pipeline will output the [generated event logs (GenBaselineED)](data/event_logs/GenBaselineED), which optimize their feature values towards [real-world event data features](data/BaselineED_feat.csv), alongside their respectively measured [feature values](data/GenBaselineED_feat.csv) and [benchmark metrics values](data/GenBaselineED_bench.csv).

```console
conda activate gedi_streams
python main.py -a config_files/experiment_real_targets.json
```

### Generating data with grid targets
We employ the [experiment_grid_2obj_configfiles_fabric.ipynb](notebooks/experiment_grid_2obj_configfiles_fabric.ipynb) to create all necessary [configuration](config_files/grid_2obj) and [objective](data/grid_2obj) files for this experiment.
To create configuration files for grid objectives interactively, you can use the start the [iGEDI Webapp](https://huggingface.co/spaces/andreamalhera/gedi) or the following dashboard from [iGEDI](https://github.com/lmu-dbs/gedi/blob/1d56290b643e3507cb0a2a70963c0e77188e045d/README.md?plain=1#L397)
For running config_files, please refer to [Feature Extraction](#feature-extraction), and [Generation](#generation).

### Visualizations
To run the visualizations, we employ [jupyter notebooks](https://jupyter.org/install) and [add the installed environment to the jupyter notebook](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084). We then start all visualizations by running e.g.: `jupyter noteboook`. In the following, we describe the `.ipynb`-files in the folder `\notebooks` to reproduce the figures from our paper. 

#### [Fig. 4 and fig. 5 Representativeness](notebooks/gedi_figs4and5_representativeness.ipynb)
To visualize the coverage of the feasible feature space of generated event logs compared to existing real-world benchmark datasets, in this notebook, we conduct a principal component analysis on the features of both settings. The first two principal components are utilized to visualize the coverage which is further highlighted by computing a convex hull of the 2D mapping.Additionally, we visualize the distribution of each meta feature we used in the paper as a boxplot. Additional features can be extracted with FEEED. Therefore, the notebook contains the figures 4 and 5 in the paper.

#### [Fig. 9 Consistency and fig. 10 Limitations](notebooks/gedi_figs9and10_consistency.ipynb)
Likewise to the evaluation on the statistical tests in notebook `gedi_figs7and8_benchmarking_statisticalTests.ipynb`, this notebook is used to compute the differences between two correlation matrices $\Delta C = C_1 - C_2$. This logic is employed to evaluate and visualize the distance of two correlation matrices. Furthermore, we show how significant scores are retained from the correlations being evaluated on real-world datasets coompared to synthesized event log datasets with real-world targets. In Fig. 9 and 10 in the paper, the results of the notebook are shown. 

## Citation