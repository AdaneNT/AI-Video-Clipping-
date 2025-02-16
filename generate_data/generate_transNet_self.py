import sys
import os

import import_ipynb
from TransNetV2_master.inference.transnetv2 import TransNetV2
from tqdm import tqdm
from generate_testData_h5.networks import ResNet
import numpy as np
import h5py
import decord
import torch
import torch.nn as nn
from decord import VideoReader

# Self-Attention module
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Compute queries, keys, and values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.shape[-1])
        attention_weights = self.softmax(scores)

        # Weighted sum of values
        attended = torch.matmul(attention_weights, V)
        return attended

class GenerateDataset:
    def __init__(self, video_path, save_path, embed_size=2048):
        self.resnet = ResNet()  # Pre-trained ResNet
        self.attention = SelfAttention(embed_size)  # Self-Attention layer
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.h5_file = h5py.File(save_path, 'w')
        self.model = TransNetV2()  # Initialize TransNetV2


        self.set_video_list(video_path)

    def set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = sorted(os.listdir(video_path))
            self.video_list = [x for x in self.video_list if x.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx + 1)] = {}
            self.h5_file.create_group('video_{}'.format(idx + 1))

    def extract_feature(self, frame):
        '''
        Extract frame feature by passing it through pre-trained ResNet and Self-Attention.
        '''
        # Extract features using ResNet
        res_pool5 = self.resnet(frame)
        
        # Add batch dimension for self-attention processing if needed
        res_pool5 = res_pool5.unsqueeze(0)  # Shape: (1, num_features)
        
        # Apply Self-Attention on the ResNet features
        attended_features = self.attention(res_pool5)
        
        # Remove batch dimension and flatten for saving
        frame_feat = attended_features.squeeze(0).cpu().data.numpy().flatten()

        return frame_feat

    def get_change_points(self, video_path):
        '''
        Extract indices of keyframes using TransNetV2 and construct segments
        '''
          # Get predictions from TransNetV2
        _, single_frame_predictions, _ = self.model.predict_video(video_path)
        shots = self.model.predictions_to_scenes(single_frame_predictions)

        # Convert scenes to change points
        change_points = np.array(shots)

        # Calculate the number of frames per segment
        n_frame_per_seg = np.array([end - start + 1 for start, end in change_points])

        return change_points, n_frame_per_seg

    def generate_dataset(self):
        '''
        Convert from video file (mp4) to h5 file with the right format
        '''
        for video_idx, video_filename in enumerate(tqdm(self.video_list, desc='Feature Extract', ncols=80, leave=True)):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_name = os.path.basename(video_path)

            vr = decord.VideoReader(video_path, width=224, height=224)  # for passing through resnet

            fps = vr.get_avg_fps()
            n_frames = len(vr)

            frame_list = []
            picks = []  ## position of heuristically-selected keyframes 
            video_feat = None

            change_points, n_frame_per_seg = self.get_change_points(video_path)

            # for representing the main features of the whole segment
            for segment in change_points:
                mid = (segment[0] + segment[1]) // 2
                frame = vr[mid].asnumpy()

                frame_feat = self.extract_feature(frame)

                picks.append(mid)

                if video_feat is None:
                    video_feat = frame_feat
                else:
                    video_feat = np.vstack((video_feat, frame_feat))

            self.h5_file['video_{}'.format(video_idx + 1)]['features'] = list(video_feat)  # frame features 
            self.h5_file['video_{}'.format(video_idx + 1)]['picks'] = np.array(list(picks))  # keyframes or segments in the video 
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames  # total number of frames in the video
            self.h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps  # frames per second (fps) of the current video
            self.h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points  # key points where significant changes occur in the video.
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg  # number of frames in each segment of video
            self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_name

        self.h5_file.close()

if __name__ == '__main__':
    pass
