from sclip_generate_video.generate_video_json_params import generate_video
from models_train.solver import Solver
from data_loader.data_loader import get_loader_custom_video_data
from configs.configs import get_config
from generate_data.generate_heuristic_self import GenerateDataset
from flask import Flask, request, jsonify, render_template
from flasgger import Swagger, swag_from
from werkzeug.utils import secure_filename
import json
import os

app = Flask(__name__)

# Swagger UI template
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "AI Video Clipping API",  # Custom title for the API
        "description": "This API generates summaries for uploaded videos. We can clip, extract, and generate summarized content.",  # Custom description
        "version": "0.1"
    },
    "tags": [
        {
            "name": "Test with PWR input",  # Tag name
            "description": "clip, extract, and generate summarized content."  # Tag description
        },
    ],
    "basePath": "/"
})

UPLOAD_FOLDER = 'uploads/'
RESULTS_FOLDER = 'results_all/'
SCORES_FOLDER = os.path.join(RESULTS_FOLDER, 'TV2/scores')
MODELS_FOLDER = os.path.join(RESULTS_FOLDER, 'TV2/models')
VIDEO_EDL_OUTPUT = os.path.join(RESULTS_FOLDER, 'Video_EDL_output')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCORES_FOLDER, exist_ok=True)
os.makedirs(VIDEO_EDL_OUTPUT, exist_ok=True)

# Predefined fixed durations for each video type
video_durations = {
    "LANGLOT": (4, 20),  # 4 minutes 20 seconds
    "LOT": (2, 40),      # 2 minutes 40 seconds
    "KORTLOT": (1, 40),  # 1 minute 30 seconds
    "LYD": (0, 40)       # 40 seconds
}

@app.route('/generate_video_clip', methods=['POST'])
@swag_from({
    'tags': ['Test with PWR input'],  # Assign a specific tag
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Video file to upload'
        },
        {
            'name': 'video_type',
            'in': 'formData',
            'type': 'string',
            'enum': ['LANGLOT', 'LOT', 'KORTLOT', 'LYD'],  # Dropdown options
            'required': True,
            'description': 'Select video type to determine clipping duration'
        },
        {
            'name': 'start_time',
            'in': 'formData',
            'type': 'integer',
            'required': True,
            'description': 'Enter the start time (seconds)'
        },
        {
            'name': 'frames',
            'in': 'formData',
            'type': 'integer',
            'required': False,
            'description': ''
        }
    ],
    'responses': {
        200: {
            'description': 'Video summary generated successfully.',
            'examples': {
                'application/json': {
                    'message': 'Summary and video generated successfully.',
                    'summary_path': 'results_all/TV2/scores/split0/custom_video_-1.json',
                    'video_path': 'results_all/Video_EDL_output/ml_example.mp4',
                    'edl_data': {
                        "tracks": [
                            {
                                "type": "video",
                                "width": 1280,
                                "height": 720,
                                "frameRate": 30,
                                "segments": [
                                    {"source": "example.mp4", "start": 10.0, "duration": 5.0}
                                ]
                            },
                            {
                                "type": "audio",
                                "sampleRate": 48000,
                                "channels": 2,
                                "segments": [
                                    {"source": "example.mp3", "start": 10.0, "duration": 5.0}
                                ]
                            }
                        ]
                    }
                }
            }
        },
        400: {
            'description': 'Invalid input or missing file.',
            'examples': {
                'application/json': {
                    'error': 'No file part or invalid input.'
                }
            }
        },
        500: {
            'description': 'Internal server error.',
            'examples': {
                'application/json': {
                    'error': 'Error while processing the video.'
                }
            }
        }
    }
})
def generate_summary():
    """
    Generate a clip for the uploaded video and create a summarized video from it.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Parse additional parameters
    video_type = request.form.get('video_type', type=str)
    start_time = request.form.get('start_time', type=int)
    frames = request.form.get('frames', type=int)
    
    # Get the duration for the selected video type
    if video_type not in video_durations:
        return jsonify({'error': 'Invalid video type selected'}), 400

    d_min, d_sec = video_durations[video_type]
    
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        try:
            save_path = os.path.join(RESULTS_FOLDER, 'test_feature.h5')

            # Feature extraction
            gen_data = GenerateDataset(video_path, save_path)
            gen_data.generate_dataset()

            # Initialize test config
            config = get_config(mode='test', video_type='TV2')

            # Initialize data loaders
            train_loader = None
            test_loader = get_loader_custom_video_data(config.mode, save_path)

            # Evaluation
            solver = Solver(config, train_loader, test_loader)
            solver.build()
            solver.loadfrom_checkpoint(os.path.join(MODELS_FOLDER, 'split0/epoch-84.pkl'))
            solver.evaluate(-1)

            # Generate video and EDL
            score_path = os.path.join(SCORES_FOLDER, f'split{config.split_index}/custom_video_-1.json')
            generate_video(score_path, save_path, video_path, d_min=d_min, d_sec=d_sec, start_time=start_time, frames=frames)

            # Read the EDL JSON output
            edl_output_path = os.path.join(VIDEO_EDL_OUTPUT, f'EDL_{filename[:-4]}.json')
            with open(edl_output_path, 'r') as edl_file:
                edl_data = json.load(edl_file)

            return jsonify({
                'message': 'Summary and video generated successfully.',
                'summary_path': score_path,
                'video_path': os.path.join(VIDEO_EDL_OUTPUT, f'ml_{filename}'),
                'edl_data': edl_data
            }), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
