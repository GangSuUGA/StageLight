import configparser
from sumolib import checkBinary
import os
import sys


def import_train_configuration(config_file):
    """
    Read the config file regarding the training and import its content
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}
    config['models_path_name'] = content['dir']['models_path_name']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config

def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode    
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
 
    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name), "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "--start", "--quit-on-end"]

    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 

def get_model_path(models_path_name, model_n):
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')
    return model_folder_path

def set_test_path(models_path_name, model_n):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_'+str(model_n), '')

    if os.path.isdir(model_folder_path):    
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else: 
        sys.exit('The model number specified does not exist in the models folder')