## Automated News Clip Generation via Robust Video Summarization
This repository provides a framework for clipping and generating summarized video content along with edit decision lists (EDLs). It involves advanced deep learning models, including Bayesian-based variational autoencoders, multi-head attention mechanisms, and contrastive learning methods, integrated with robust feature extraction components. The framework is designed to generate coherent, high-quality summaries while preserving spatial-temporal continuity across video segments.

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

## Datasets
**Newsroom dataset**:The dataset used for training the models consists of a collection of news videos (commonly named as POWERs or raw footage) from the TV2 news collection. The files are preprocessed and structured in h5py format, containing video features, segmentations, and other relevant information. These files are stored in the `data` folder and include essential information for video clipping and summary generation. 

**Other Datasets**: **TVSum** and **SumMe** for direct comparison with stateof-the-art (SOTA) methods. TVSum dataset comprises 50 videos, each typically ranging from 1 to 5 minutes in
duration. The dataset includes various genres, such as news and documentary. SumMe dataset consists of 25 videos, with durations ranging from 1 to 6 minutes. 

## How to Train
 We train the model using 5 splits of the dataset. You can train the model by running the following command:

```bash
python train.py  /path/to/dataset
 ```
## How to Test (Generate Clip)
Generate a clip (a short version of the original footage) using the graphical user interface (GUI). The GUI lets you choose a clip type—`LYD`, `KORTLOT`, `LOT`, or `LANGLOT`—to determine the clip length. An Edit Decision List (EDL) can also be generated automatically for each selected clip type.

- **LYD (shortest)**: 00:10–00:39  
- **KORTLOT (short)**: 00:40–01:14  
- **LOT (medium)**: 01:15–02:29  
- **LANGLOT (long)**: 02:30–04:45-

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
The codebase is based upon the following repositories:
- [Adversarial Video Summary](https://github.com/j-min/Adversarial_Video_Summary).
- [AC-SUM-GAN](https://github.com/e-apostolidis/AC-SUM-GAN)

We casted our video clipping problem as extractive video summarization task on top of the existing methods, which have been extended and modified to include additional learning techniques such as Bayesian Variational Autoencoders, transformers, and novel contrastive loss for enhanced clipping and maintaining spatial-temporal continuity across shots.


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

   





