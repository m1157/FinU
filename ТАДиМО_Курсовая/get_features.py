import librosa
import re
import numpy as np
import os
import json

def get_song_name(path, n_slice=None, file_format=True):
    song_name = path.replace('/', '\\').split('\\')[-1:][0]
    song_name_words = song_name.split('.wav')
    if n_slice == None:
        return song_name_words[0] + '.wav' * file_format
    else:
        return song_name_words[0] + '_part_' + str(n_slice + 1) + '.wav' * file_format

    
def get_song_genre(path):
    path = path.replace('/', '\\')
    return re.compile('.*\\\\(.*)\..*.wav').findall(path)[0]


def get_song_features_set_1(path):
    y, sr = librosa.load(path)
    features = dict()
    
    features['spectral_centroid_mean'] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features['spectral_centroid_std'] = librosa.feature.spectral_centroid(y=y, sr=sr).std()
    features['spectral_rolloff_mean'] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    features['spectral_rolloff_std'] = librosa.feature.spectral_rolloff(y=y, sr=sr).std()
    features['spectral_bandwidth_mean'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    features['spectral_bandwidth_std'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).std()
    features['spectral_flux_mean'] = librosa.onset.onset_strength(y=y, sr=sr).mean()
    features['spectral_flux_std'] = librosa.onset.onset_strength(y=y, sr=sr).std()    
    features['zero_crossings_mean'] = librosa.feature.zero_crossing_rate(y=y).mean()
    features['zero_crossings_std'] = librosa.feature.zero_crossing_rate(y=y).std()
    features['tempo'] = librosa.feature.tempo(y=y)[0]

    rms = librosa.feature.rms(y=y)
    threshold = np.mean(rms)
    low_energy = sum(rms[0] < threshold) / len(rms[0])
    features['low_energy'] = low_energy

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for i in range(5):
        features['mfcc_' + str(i+1) + '_mean'] = mfcc[i].mean()
        features['mfcc_' + str(i+1) + '_std'] = mfcc[i].std()

    return features


def get_song_features_set_2(path, n_slices):
    duration = 29
    y, sr = librosa.load(path, duration=duration)
    track_features = dict()
    samples_per_slice = int(duration * sr / n_slices)

    for slice_ in range(n_slices):
        start_sample = samples_per_slice * slice_
        end_sample = start_sample + samples_per_slice
        y_slice = y[start_sample:end_sample]
        
        features = dict()
        features['spectral_centroid_mean'] = librosa.feature.spectral_centroid(y=y_slice, sr=sr).mean()
        features['spectral_centroid_std'] = librosa.feature.spectral_centroid(y=y_slice, sr=sr).std()
        features['spectral_rolloff_mean'] = librosa.feature.spectral_rolloff(y=y_slice, sr=sr).mean()
        features['spectral_rolloff_std'] = librosa.feature.spectral_rolloff(y=y_slice, sr=sr).std()
        features['spectral_bandwidth_mean'] = librosa.feature.spectral_bandwidth(y=y_slice, sr=sr).mean()
        features['spectral_bandwidth_std'] = librosa.feature.spectral_bandwidth(y=y_slice, sr=sr).std()
        features['spectral_flux_mean'] = librosa.onset.onset_strength(y=y_slice, sr=sr).mean()
        features['spectral_flux_std'] = librosa.onset.onset_strength(y=y_slice, sr=sr).std()   
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
        song_name = get_song_name(path, slice_)
        
        track_features[song_name] = features
    return track_features


def get_song_features_set_3(source_path, json_path):
    mydict = {"labels": [], "features": []}
    duration = 29
    n_slices = 10
   
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(source_path)):
        for file in filenames:
            song, sr = librosa.load(os.path.join(dirpath, file), duration=duration)
            samples_per_slice = int(duration * sr / n_slices)
            for s in range(n_slices):
                start_sample = samples_per_slice * s
                end_sample = start_sample + samples_per_slice
                mfcc = librosa.feature.mfcc(y=song[start_sample:end_sample], sr=sr, n_mfcc=5)
                mfcc = mfcc.T
                mydict["labels"].append(i-1)
                mydict["features"].append(mfcc.tolist())
   
    with open(json_path, 'w') as f:
        json.dump(mydict, f)
    f.close()
