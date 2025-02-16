import argparse
from sclip_generate_video.generate_video_json import generate_video
from models_train.solver import Solver
from data_loader.data_loader import get_loader_custom_video_data
from configs.configs import get_config
from generate_data.generate_heuristic_self import GenerateDataset

def main(args):
    video_path = args.video_path
    d_min = args.d_min
    d_sec = args.d_min
    start_time = args.start_time
    frames = args.frames

    save_path = 'results_all/test_feature.h5'

    # Feature extraction
    gen_data = GenerateDataset(video_path, save_path)
    gen_data.generate_dataset()

    # Initialize test configuration
    config = get_config(mode='test', video_type='TV2')
    print(config)

    # Initialize data loader
    train_loader = None
    test_loader = get_loader_custom_video_data(config.mode, save_path)

    # Evaluation
    solver = Solver(config, train_loader, test_loader)
    solver.build()
    solver.loadfrom_checkpoint('./results_all/TV2/models/split0/epoch-84.pkl')
    solver.evaluate(-1)

    # Generate video
    score_path = f'results_all/TV2/scores/split{config.split_index}/TV2_-1.json'
    generate_video(score_path, save_path, video_path, d_min,d_sec,start_time, frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for video summarization")
    
    # Required arguments
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--d_min", type=int, required=True, help="Desired clip duration in minutes")
    parser.add_argument("--d_sec", type=int, required=True, help="Desired clip duration in minutes")
    parser.add_argument("--start_time", type=int, required=True, help="Start time in seconds")
    parser.add_argument("--frames", type=int, required=True, help=" number of frames")

    # Optional arguments
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()
    main(args)
