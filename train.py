import sys
from models_train.solver import Solver
from data_loader.data_loader import get_loader,get_loader_custom_video_data
from configs.configs import get_config
import os

if __name__ == '__main__':
    # Check the path 
    if len(sys.argv) > 1:
        use_custom_video = True  # 
        data_path = sys.argv[1].strip()  #

        # Check if the file exists
        if not os.path.exists(data_path):
            print(f"Error: The dataset file '{data_path}' was not found.")
            sys.exit(1)
    else:
        use_custom_video = False  # 

    if use_custom_video:
        # Initialize training config
        config = get_config(mode='train', video_type='TV2')
        test_config = get_config(mode='test', video_type='TV2')
        print(config)
        print(test_config)

        # Initialize data loader 
        train_loader = get_loader_custom_video_data(config.mode, data_path)
        test_loader = get_loader_custom_video_data(test_config.mode, data_path)

        # Train
        solver = Solver(config, train_loader, test_loader)
        solver.build()
        solver.train()
    else:
        # Initialize configs for default dataset (e.g., 'tvsum')
        config = get_config(mode='train')
        test_config = get_config(mode='test')
        print(config)
        print(test_config)

        # Initialize data loader for default dataset
        train_loader = get_loader('tvsum', config.mode, config.split_index)
        test_loader = get_loader('tvsum', test_config.mode, test_config.split_index)

        # Evaluate and train
        solver = Solver(config, train_loader, test_loader)
        solver.build()
        solver.evaluate(-1)  # 
        solver.train()
