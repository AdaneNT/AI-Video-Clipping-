## Video Clipping and Summary Generation Framework
This repository provides a comprehensive framework for clipping, extracting, and generating summarized video content along with edit decision lists (EDLs). It involves advanced deep learning models, including  variational autoencoders, multi-head attention mechanisms, and long short-term memory (LSTM) networks, integrated with robust feature extraction components. We extend the existing work by adding new components including  Bayesian Inference, transformers, and contrastive learning for enhanced clipping and maintaining spatial-temporal continuity across segments.
## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [How to Train](#how-to-train)
- [How to Test (Generate Clip)](#how-to-test-generate-clip)
- [Acknowledgements](#acknowledgements)
- [References](#references)

## Requirements
```bash
pip install -r requirements.txt
 ```  

## Dataset
The dataset used for training the models consists of a collection of news videos (commonly named as POWERs or raw footage) from the **TV2 News** collection. The files are preprocessed and structured in h5py format, containing video features, segmentations, and other relevant information. These files are stored in the `data` folder and include essential information for video clipping and summary generation.  

## How to Train
This work is ongoing, and we train the model using a single split of the dataset. You can train the model by running the following command:

```bash
python train.py  /path/to/dataset
 ```
## How to Test (Generate Clip)
Generate a clip (short version of the orginal footage) using graphical user Interface (GUI). The GUI allows the user to choose a clip type (LYD, KORTLOT,LOT, and LANGLOT) to determine the length of the clip. An Edit Decision List (EDL) can also be automatically generated for each selected clip type.
- **LYD**: 10-40 sek      
- **KORTLOT**: 40 sek - 1 min 30 sek
- **LOT**: 1 min 15 sek - 2 min 45 sek
- **LANGLOT**: 2 min 30 sek - 4 min 45 sek

### Using Swagger UI:
1. Launch the API by running the following command:
   ```bash
   python test_api_params.py
    ```
2. Open your web browser and navigate to the Swagger UI at http://localhost:8080/apidocs/ (or the address specified in the terminal).
3. Use the Swagger UI interface to upload your video and set parameters to generate a clip.
   
### Command-Line Usage:
Alternatively, you can generate a clip by running the following command in the terminal:
```bash
python test.py --video_path /path/to/video --d_min 2 --d_sec 30 --start_time  50 --frames 0
 ```
- --video_path <str>:  Path to the input video file.
- --d_min <int>:  Desired clip duration in minutes.
- --d_sec <int>: Desired clip duration in seconds.
- --s_time <int>:  Start time in seconds (optional, default :50)
- --frames <int>:  Number frames (optional, default :0 )

## Acknowledgements
The code implementation is based upon the following related repositories:
- [Adversarial Video Summary](https://github.com/j-min/Adversarial_Video_Summary) by j-min, which serves as the  building block for this project.
- [AC-SUM-GAN](https://github.com/e-apostolidis/AC-SUM-GAN) by e-apostolidis

We casted our video clipping problem as extractive video summarization task on top of the existing methods, which have been extended and modified to include additional learning techniques such as Bayesian Variational Autoencoders, transformers, and contrastive learning for enhanced clipping and maintaining spatial-temporal continuity across shots.

## References
1. **Unsupervised Video Summarization with Adversarial LSTM Networks**  
   Behrooz Mahasseni, Michael Lam, and Sinisa Todorovic.  
2. **AC-SUM-GAN: Connecting Actor-Critic and Generative Adversarial Networks for Unsupervised Video Summarization**  
   Evlampios Apostolidis, Eleni Adamantidou, Alexandros I. Metsa
3. **Learning to Cut by Watching Movies**  
   Alejandro Pardo, Fabian Caba Heilbron, Juan Leon Alcázar, Ali Thabet, Bernard Ghanem.  
4. **AI Video Editing: a Survey**, Xinrong Zhang, Yanghao Li2, Yuxing Han, Jiangtao Wen
5. **Self-Attention Based Generative Adversarial Networks for Unsupervised Video Summarization**  
   Maria Nektaria Minaidi, Charilaos Papaioannou, Alexandros Potamianos.  
6. **AI-Based Video Clipping of Soccer Events**,
   Joakim Olav Valand et al.
7. **Video Summarization with Long Short-term Memory**
  Ke Zhang, Wei-Lun Chao, Fei Sha, Kristen Grauman.

   





