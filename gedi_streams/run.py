import config
import pandas as pd
from datetime import datetime as dt
from gedi_streams.generator.generator import GenerateEventLogs
from gedi_streams.features.feature_extraction import EventLogFeatures
from gedi_streams.plotter import FeaturesPlotter, GenerationPlotter
from gedi_streams.utils.param_keys import *
from utils.default_argparse import ArgParser

def run(kwargs:dict, model_params_list: list, filename_list:list):
    """
    This function chooses the running option for the program.
    @param kwargs: dict
        contains the running parameters and the event-log file information
    @param model_params_list: list
        contains a list of model parameters, which are used to analyse this different models.
    @param filename_list: list
        contains the list of the filenames to load multiple event-logs
    @return:
    """
    params = kwargs[PARAMS]
    ft = EventLogFeatures(None)
    gen = pd.DataFrame(columns=['log'])

    for model_params in model_params_list:
        if model_params.get(PIPELINE_STEP) == 'event_logs_generation':
            gen = pd.DataFrame(GenerateEventLogs(model_params).log_config)
            #gen = pd.read_csv("output/features/generated/grid_2objectives_enseef_enve/2_enseef_enve_feat.csv")
            #GenerationPlotter(gen, model_params, output_path="output/plots")
        elif model_params.get(PIPELINE_STEP) == 'feature_extraction':
            ft = EventLogFeatures(**kwargs, logs=gen['log'], ft_params=model_params)
            FeaturesPlotter(ft.feat, model_params)
        elif model_params.get(PIPELINE_STEP) == "evaluation_plotter":
            GenerationPlotter(gen, model_params, output_path=model_params['output_path'], input_path=model_params['input_path'])

def gedi(config_path):
    """
    This function runs the GEDI pipeline.
    @param config_path: str
        contains the path to the config file
    @return:
    """
    model_params_list = config.get_model_params_list(config_path)
    run({'params':""}, model_params_list, [])
