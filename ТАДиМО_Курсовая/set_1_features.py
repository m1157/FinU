import librosa
import re
import numpy as np

def get_song_name(path, n_of_slice):
    """Returns the name of the file located in path"""
    song_name = path.replace('/', '\\').split('\\')[-1:][0]
    song_name_words = song_name.split('.wav')
    
    return song_name_words[0] + '_part_' + str(n_of_slice + 1) + '.wav'

def get_song_genre(path):
    """Returns the genre name of the file located in path"""
    path = path.replace('/', '\\')
    try:
        return re.compile('.*\\\\(.*)\..*.wav').findall(path)[0]
    except:
        return '–'
    else:
        return re.compile('.*\\\\(.*)\..*.wav').findall(path)[0]
    
def get_song_features(path, test=False):
    """
    Extracting .wav file characteristics 
    Return value: 
        song_name (str): file name
        features (dict): dictionary of calculated characteristics
    """
    num_slices = 3
    if test==False:
        duration = 29
        y, sr = librosa.load(path, duration=duration)
    else:
        y, sr = librosa.load(path)
        
    track_features = dict()
    samples_per_slice = int(duration * sr / num_slices)

    for slice in range(num_slices):
        start_sample = samples_per_slice * slice
        end_sample = start_sample + samples_per_slice
        y_slice = y[start_sample:end_sample]
        
        features = dict()
        features['spectral_centroid_mean'] = librosa.feature.spectral_centroid(y=y_slice, sr=sr).mean()
        features['spectral_centroid_std'] = librosa.feature.spectral_centroid(y=y_slice, sr=sr).std()
        features['spectral_rolloff_mean'] = librosa.feature.spectral_rolloff(y=y_slice, sr=sr).mean()
        features['spectral_rolloff_std'] = librosa.feature.spectral_rolloff(y=y_slice, sr=sr).std()
        features['spectral_bandwidth_mean'] = librosa.feature.spectral_bandwidth(y=y_slice, sr=sr).mean()
        features['spectral_bandwidth_std'] = librosa.feature.spectral_bandwidth(y=y_slice, sr=sr).std()
        features['zero_crossings_mean'] = librosa.feature.zero_crossing_rate(y=y_slice).mean()
        features['zero_crossings_std'] = librosa.feature.zero_crossing_rate(y=y_slice).std()
        features['tempo'] = librosa.feature.tempo(y=y_slice)[0]

        rms = librosa.feature.rms(y=y_slice)
        threshold = np.mean(rms)
        low_energy = sum(rms[0] < threshold) / len(rms[0])
        features['low_energy'] = low_energy

        mfcc = librosa.feature.mfcc(y=y_slice, sr=sr)
        for i in range(5):
            features['mfcc_' + str(i+1) + '_mean'] = mfcc[i].mean()
            features['mfcc_' + str(i+1) + '_std'] = mfcc[i].std()
        
        features['genre'] = get_song_genre(path)
        song_name = get_song_name(path, slice)
        
        track_features[song_name] = features
    return track_features
