from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation 
from generator import TrafficGenerator
from memory import Memory
from model import Train_Model
from visualization import Visualization
from _utils import import_train_configuration, set_sumo, set_train_path,get_model_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])
    model_path = get_model_path(config['models_path_name'], config['model_to_test'])

    Model = Train_Model(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions'],
        tau = 1
    )
    
    # Model_B = TrainModel_B(
    #     config['num_layers'], 
    #     config['width_layers'], 
    #     config['batch_size'], 
    #     config['learning_rate'], 
    #     input_dim=config['num_states'], 
    #     output_dim=config['num_actions']
    # )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        config['batch_size']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
     
    # Model._load_my_model(model_path)
    # Model._load_my_target_model(model_path)
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = (1.0 - (episode / config['total_episodes']))  # set the epsilon for this episode according to epsilon-greedy policy
        # epsilon = 0
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1
        # print(Simulation.loss_store)
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model(path)
    # Model.save_target_model(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.speed_store, filename='Speed', xlabel='Episode', ylabel='Average Speed')  
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation.loss_store, filename='Loss', xlabel='Episode', ylabel='Total Loss')