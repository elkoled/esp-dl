import librosa
import numpy as np
import os
import random

def load_and_preprocess(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    
    # Ensure the audio is 1s
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    else:
        y = y[:16000]
    
    return y

base_directory = '/mnt/d/Repositories/KWS_Streaming/data2_backup'  # Modify this to the directory containing the folders
folders = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'marvin']

data = []

for folder in folders:
    folder_path = os.path.join(base_directory, folder)
    
    # Get all wav files from the folder
    all_wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    # Randomly sample 100 wav files
    sampled_wav_files = random.sample(all_wav_files, 100)
    
    for file in sampled_wav_files:
        audio_data = load_and_preprocess(file)
        data.append(audio_data)
        break
    break

# Convert the data list to a numpy array
data_np = np.array(data)

# Save the data to a .npy file
np.save('output_data_1.npy', data_np)
