import sys
import os
import json
import numpy as np
import h5py
import decord
import cv2
import moviepy.editor as mpe
from pydub import AudioSegment
from sclip_generate_scores.generate_scores import generate_summary

def generate_video(score_path, metadata_path, video_path, d_min,d_sec, start_time, frames):
    # Generate summaries
    fps=30
    all_scores = []
    with open(score_path) as f:
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:
            scores = np.asarray(data[video_name])
            all_scores.append(scores)

    all_shot_bound, all_nframes, all_positions = [], [], []
    with h5py.File(metadata_path, 'r') as hdf:
        for video_key in keys:
            video_index = video_key[6:]  # Extract video index from key
            
            sb = np.array(hdf.get('video_' + video_index + '/change_points'))
            n_frames = np.array(hdf.get('video_' + video_index + '/n_frames'))
            positions = np.array(hdf.get('video_' + video_index + '/picks'))

            all_shot_bound.append(sb)
            all_nframes.append(n_frames)
            all_positions.append(positions)

    all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions,d_min,d_sec)

    # Generate video
    with h5py.File(metadata_path, 'r') as hdf:
        for video_index, video_key in enumerate(keys):  # Use enumerate to get both index and key
            summary = all_summaries[video_index]  # Get the summary for the current video
            
            # Extract metadata
            video_name = hdf[video_key + '/video_name'][()].decode()
            audio_name = video_name[:-4] + '.mp3'   # Change the extension from mp4 to mp3

            
            if os.path.isdir(video_path):
                tmp_path = os.path.join(video_path, video_name)
            else:
                tmp_path = video_path

            video_reader = decord.VideoReader(tmp_path)
            audio_reader = AudioSegment.from_file(tmp_path, 'mp4')

            fps = video_reader.get_avg_fps()
            (frame_height, frame_width, _) = video_reader[0].asnumpy().shape

            # check start
            timecode = (fps * start_time) + frames
            frames_to_skip = int(timecode)
            summary[:frames_to_skip] = 0  # Set first 57 seconds in the summary to 0

             
            summary[frames_to_skip:frames_to_skip + int(fps * 5)] = 1
            #summary[-int(fps * 5):] = 1

            # Extract frame indices
            frame_numbers = list(np.argwhere(summary == 1).reshape(1, -1).squeeze(0))

            # Create segments for EDL based on selected shots
            shot_bound = all_shot_bound[video_index]  # Access the correct shot boundaries using index

            # Prepare the EDL structure
            edl = {
                "tracks": [
                    {
                        "type": "video",
                        "width": frame_width,
                        "height": frame_height,
                        "frameRate": fps,
                        "segments": []
                    },
                    {
                        "type": "audio",
                        "sampleRate": 48000,
                        "channels": 2,
                        "segments": []
                    }
                ]
            }

            # Iterate over each shot in shot_bound and add to EDL if it overlaps with non-zero summary frames
            for shot in shot_bound:
                shot_start, shot_end = shot[0], shot[1]

                # Check if the shot is included in the summary
                if np.any((frame_numbers >= shot_start) & (frame_numbers <= shot_end)):
                    
                    # Find the first frame in summary that is included for this shot
                    start_frame = np.argwhere(summary[shot_start:shot_end + 1] == 1)[0][0] + shot_start
                    start_time = start_frame / fps  # Convert to seconds

                    # Calculate duration based on the shot's end frame and the summary frames
                    duration = (shot_end - shot_start + 1) / fps  # Duration in seconds
                    
                    # Add video segment
                    edl["tracks"][0]["segments"].append({
                        "source": video_name,
                        "start": start_time,
                        "duration": duration
                    })
                    
                    # Add audio segment
                    edl["tracks"][1]["segments"].append({
                        "source": audio_name,
                        "start": start_time,
                        "duration": duration
                    })

                    # show segments 
                    print(f"Adding segment: video_start={start_time}, video_duration={duration} for shot: {shot}")
            
            # Check if any segments were added to the EDL
            if not edl["tracks"][0]["segments"]:
                print("Warning: No segments added to EDL.")

            # Write the EDL to a JSON file
            edl_file_path = 'results_all/Video_EDL_output/' + "EDL_" + video_name[:-4] + '.json'
            with open(edl_file_path, 'w') as edl_file:
                json.dump(edl, edl_file, indent=4)

            # Video writing part remains unchanged
            vid_writer = cv2.VideoWriter(
                'results_all/Video_EDL_output/' + video_name,
                cv2.VideoWriter_fourcc(*'MP4V'),
                fps,
                (frame_width, frame_height)
            )
            summarized_audio = None

            # Generate summary video and audio
            for idx in frame_numbers:
                if idx >= frames_to_skip:  # Only write frames after skipping initial frames
                    # Write video to file
                    frame = video_reader[idx]
                    vid_writer.write(cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))

                    # Get audio index in seconds
                    audio_start_idx, audio_end_idx = video_reader.get_frame_timestamp(idx)
                    # Seconds to milliseconds
                    audio_start_idx = round(audio_start_idx * 1000)
                    audio_end_idx = round(audio_end_idx * 1000)

                    # Concatenate audio segment
                    if summarized_audio is None:
                        summarized_audio = audio_reader[audio_start_idx:audio_end_idx]
                    else:
                        summarized_audio += audio_reader[audio_start_idx:audio_end_idx]

            # Write audio to file
            summarized_audio.export('results_all/Video_EDL_output/' + audio_name, format='mp3')

            vid_writer.release()

            # Combine video and audio
            input_video = mpe.VideoFileClip('results_all/Video_EDL_output/' + video_name)
            input_audio = mpe.AudioFileClip('results_all/Video_EDL_output/' + audio_name)

            output_video = input_video.set_audio(input_audio)
            
            output_video.write_videofile(
                'results_all/Video_EDL_output/ml_' + video_name,
                codec='libx264',
                audio_codec='aac',
            )

            # Clean up temporary files
            os.remove('results_all/Video_EDL_output/' + video_name)
            os.remove('results_all/Video_EDL_output/' + audio_name)

if __name__ == "__main__":
    pass

