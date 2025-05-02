import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vad import EnergyVAD
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
from scipy.fft import fft, ifft
import parselmouth
import numpy as np
import os
from tqdm import tqdm
import csv
import noisereduce as nr

# voice activity mask - to keep only voiced parts (step 1 preprocessing) ---lw el sot waty awy by3tbro unvoiced?

# Reshape to (1, num_samples) for compatibility with EnergyVAD
def apply_vad(audio, sr):
    audio_2d = np.expand_dims(audio, axis=0)

    # Initialize VAD
    vad = EnergyVAD(
        sample_rate=16000,
        frame_length=25,
        frame_shift=20,
        energy_threshold=0.05,
        pre_emphasis=0.95,
    )

    # Get voice activity (1D boolean array per frame)
    voice_activity = vad(audio_2d)

    # Apply VAD to get speech-only waveform
    speech_signal = vad.apply_vad(audio_2d).flatten()

    # Frame-based time axis for VAD output
    frame_shift_samples = int(sr * (vad.frame_shift / 1000.0))
    frame_times = np.arange(len(voice_activity)) * frame_shift_samples / sr

    return speech_signal
# denoising (step 2 preprocessing) --so far m4 bt3ml haga 

# Wavelet denoising function
def wavelet_denoise(signal, wavelet='db8', level=4, threshold_scale=0.04):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise std dev
    threshold = threshold_scale * sigma

    denoised_coeffs = [
        pywt.threshold(c, threshold, mode='soft') if i > 0 else c
        for i, c in enumerate(coeffs)
    ]
    return pywt.waverec(denoised_coeffs, wavelet)

# Pre-emphasis (step 3 preprocessing) enhancing the SNR. less used in speech recognition processing. --n4elha?

def pre_emphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def framing(denoised_audio, frame_length, hop_length):

    # Framing (2D array: shape = [frame_length, num_frames])
    return librosa.util.frame(denoised_audio, frame_length=frame_length, hop_length=hop_length)
# Apply Hamming window
def windowing(frames, frame_length):
    window = np.hamming(frame_length)
    windowed_frames = frames * window[:, np.newaxis]
    return windowed_frames
removed_counts = 0
## combining preprocessing steps
def denoise(signal,sr):
    return nr.reduce_noise(y=signal, sr=sr)
def preprocess_audio(raw_audio, sr, frame_length, hop_length):

    # Step 1: VAD
    normalization_factor = np.max(np.abs(raw_audio))
    # Normalize the audio 
    raw_audio /= normalization_factor
    speech_signal = apply_vad(raw_audio, sr)

    if speech_signal is None or len(speech_signal) == 0:
        print("VAD removed all audio; skipping.")
        removed_counts += 1
        return None, None, None
    speech_signal *= normalization_factor  # Rescale to original amplitude
    # Step 2: Denoising
    denoised_audio = denoise(speech_signal, sr)

    # Step 3: Pre-emphasis
    y_preemphasized = pre_emphasis(denoised_audio)

    if(len(denoised_audio) < frame_length):
        return None, None, None
    
    # Step 4: Framing
    frames = framing(denoised_audio, frame_length, hop_length)

    # Step 5: Windowing
    windowed_frames = windowing(frames, frame_length)

    # Step 6: Normalization (optional)
    normalized_frames = windowed_frames / np.max(np.abs(windowed_frames))

    return normalized_frames, windowed_frames, denoised_audio

## features
# STE - feature?

def short_time_energy_from_windowed(windowed_frames):
    energy = np.sum(windowed_frames ** 2, axis=0)
    return energy

def short_time_energy(y, sr, windowed_frames):
    # STE calculation
    energy = short_time_energy_from_windowed(windowed_frames)

    return {
        'mean_energy': np.mean(energy),
        'std_energy': np.std(energy),
        'max_energy': np.max(energy),
        'min_energy': np.min(energy),
        'energy_variance': np.var(energy),
    }
# MFCC 13 coeff wel delta wel del2 brdo 
# ----- 1. MFCCs -----
def calculate_mfcc(denoised_audio, sr):
    mfcc = librosa.feature.mfcc(y=denoised_audio, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    # ----- 2. Delta MFCCs -----
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_means = np.mean(delta_mfcc, axis=1)
    delta_stds = np.std(delta_mfcc, axis=1)

    # ----- 3. Delta-Delta MFCCs -----
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_means = np.mean(delta2_mfcc, axis=1)
    delta2_stds = np.std(delta2_mfcc, axis=1)

    # -- kurtosis and skewness --
    mfcc_kurtosis = kurtosis(mfcc, axis=1)
    mfcc_skewness = skew(mfcc, axis=1)

    return {
        'mfcc_means': mfcc_means,
        'mfcc_stds': mfcc_stds,
        'delta_means': delta_means,
        'delta_stds': delta_stds,
        'delta2_means': delta2_means,
        'delta2_stds': delta2_stds,
        'mfcc_kurtosis': mfcc_kurtosis,
        'mfcc_skewness': mfcc_skewness
    }

# Assume windowed_frames already exists from your framing+windowing step
def calculate_pitch_and_cepstrum(windowed_frames):
    mid_frame_idx = windowed_frames.shape[1] // 2
    windowed_frame = windowed_frames[:, mid_frame_idx]

    # FFT → log magnitude spectrum
    spectrum = np.fft.fft(windowed_frame)
    log_magnitude = np.log(np.abs(spectrum) + 1e-10)

    # IFFT to get cepstrum
    cepstrum = np.fft.ifft(log_magnitude).real

    # Define pitch range (quefrency range)
    min_quefrency = int(sr / 400)  # ~400 Hz (high pitch)
    max_quefrency = int(sr / 60)   # ~60 Hz (low pitch)

    # Find the peak in that range
    pitch_quefrency = np.argmax(cepstrum[min_quefrency:max_quefrency]) + min_quefrency
    pitch_period = pitch_quefrency / sr
    pitch_frequency = 1.0 / pitch_period

    # Output pitch
    #print(f"Estimated Pitch: {pitch_frequency:.2f} Hz")
    return pitch_frequency, cepstrum, windowed_frame

def compute_cpp_from_windowed(windowed_frames, sr):
    cpp_list = []

    for i in range(windowed_frames.shape[1]):
        frame = windowed_frames[:, i]  # one windowed frame

        spectrum = np.abs(fft(frame))**2
        log_spectrum = np.log(spectrum + 1e-10)
        cepstrum = np.real(ifft(log_spectrum))

        # Quefrency range for pitch (approx. 2 ms to 12.5 ms = 80 Hz to 500 Hz)
        quefrency_range = np.arange(int(sr / 500), int(sr / 80))
        peak_val = np.max(cepstrum[quefrency_range])
        peak_idx = np.argmax(cepstrum[quefrency_range]) + int(sr / 500)

        # Linear regression over the initial part of the cepstrum (trend)
        linear_range = cepstrum[:peak_idx]
        x = np.arange(len(linear_range))
        poly = np.polyfit(x, linear_range, 1)
        trend = np.polyval(poly, peak_idx)

        # CPP = peak - trend
        cpp = peak_val - trend
        cpp_list.append(cpp)

    return np.mean(cpp_list)

def calculate_duration(y, sr):
    duration = len(y) / sr
    return duration

def calculate_wps(duration, row):
    # Count the number of words in the sentence
    word_count = len(row['sentence'].split())
    wps = word_count / duration
    return wps

def calculate_f0_features(y, sr):
    snd = parselmouth.Sound(y, sr)
    pitch = snd.to_pitch()
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  # remove unvoiced
    
    return {
        'mean': np.mean(f0),
        'std': np.std(f0),
        '5_percentile': np.percentile(f0, 5),
        '95_percentile': np.percentile(f0, 95)
    }
def calculate_tempo(y, sr, duration):
    # Compute the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Detect onsets (indicating the start of words or syllables)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

    speech_rate = len(onsets) / duration  # Onsets per second

    return speech_rate


def calculate_formants(y, sr):
    # Create a Parselmouth Sound object
    sound = parselmouth.Sound(y, sr)

    # Perform formant analysis (for the first 5 formants)
    formant = sound.to_formant_burg()
    
    # Get the formant frequencies at the mid-point of the signal
    time = len(y) / sr / 2  # Mid-point of the signal
    formant_freqs = [formant.get_value_at_time(i, time) for i in range(1, 4)]  # First 3 formants
    
    return formant_freqs
def extract_features_from_audio(y, sr, row, win_frames):
    features = {}

    # Duration
    duration = calculate_duration(y, sr)
    features['duration'] = duration

    # WPS
    features['wps'] = calculate_wps(duration, row)  

    # F0
    f0_features = calculate_f0_features(y, sr)
    features['f0_mean'] = f0_features['mean']
    features['f0_std'] = f0_features['std']
    features['f0_5_percentile'] = f0_features['5_percentile']
    features['f0_95_percentile'] = f0_features['95_percentile']

    # Tempo
    features['tempo'] = calculate_tempo(y, sr, duration)

    # Formants
    f1, f2, f3 = calculate_formants(y, sr)
    features['formant1'] = f1
    features['formant2'] = f2
    features['formant3'] = f3

    # MFCCs
    mfcc_features = calculate_mfcc(y, sr)

    for i, val in enumerate(mfcc_features['mfcc_means']):
        features[f'mfcc_{i}_mean'] = val
    for i, val in enumerate(mfcc_features['mfcc_stds']):
        features[f'mfcc_{i}_std'] = val

    for i, val in enumerate(mfcc_features['delta_means']):
        features[f'delta_mfcc_{i}_mean'] = val
    for i, val in enumerate(mfcc_features['delta_stds']):
        features[f'delta_mfcc_{i}_std'] = val

    for i, val in enumerate(mfcc_features['delta2_means']):
        features[f'delta2_mfcc_{i}_mean'] = val
    for i, val in enumerate(mfcc_features['delta2_stds']):
        features[f'delta2_mfcc_{i}_std'] = val
    for i, val in enumerate(mfcc_features['mfcc_kurtosis']):
        features[f'mfcc_{i}_kurtosis'] = val
    for i, val in enumerate(mfcc_features['mfcc_skewness']):
        features[f'mfcc_{i}_skewness'] = val

    # Cepstrum
    cpp_mean = compute_cpp_from_windowed(win_frames, sr)
    features['cpp_mean'] = cpp_mean

    # Short-Time Energy
    ste_features = short_time_energy(y, sr, win_frames)
    features['ste_mean'] = ste_features['mean_energy']
    features['ste_std'] = ste_features['std_energy']
    features['ste_max'] = ste_features['max_energy']
    features['ste_min'] = ste_features['min_energy']
    features['ste_variance'] = ste_features['energy_variance']
    
    return features
def read_data(file_path, audio_dir):
# Load your TSV
    df = pd.read_csv(file_path, sep="\t")

    available_files = set(os.listdir(audio_dir))

    # Extract base filenames from 'path' column
    df['filename'] = df['path'].apply(lambda x: os.path.basename(x))




    # Filter rows to only those where the file exists
    df = df[df['filename'].isin(available_files)]

    # Rebuild full path now that we’ve filtered
    # Normalize full path to avoid mixed slashes
    df['full_path'] = df['filename'].apply(lambda x: os.path.normpath(os.path.join(audio_dir, x)))

    return df
def load_audios(df):
    # reading data => keeping 1k males and 1k females with 0 downvotes and duration > ?
    durations = []
    # list to keep y,sr then append them to the df
    y_list = []
    sr_list = []
    print("Calculating durations...")
    for path in tqdm(df['full_path'], desc="Processing"):
        try:
            y, sr = librosa.load(path, sr=None)
            y_list.append(y)
            sr_list.append(sr)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)
        except Exception as e:
            #print(f"⚠️ Error with file {path}: {e}")
            y_list.append(None)
            sr_list.append(None)
            durations.append(None)

    df['duration'] = durations
    df['y'] = y_list
    df['sr'] = sr_list
    return df
def filter_data(df):
    duration_threshold = 2.0  
    
    # Step 1: Filter by down_votes and duration
    df_filtered = df[ (df['duration'] > duration_threshold)]
    
    # return df_filtered[(df_filtered['age'] == 'twenties') & (df_filtered['gender'] == 'female')]
    return df_filtered[(df_filtered['age'] == 'fifties')]
    # return df_filtered[(df_filtered['age'] == 'twenties') & (df_filtered['gender'] == 'male')].sample(n=1200)


def preprocess_audio_batch(df_final):    
    # preprocess and extract features for the final dataset
    # loop on data fram [y] and [sr] columns and apply the function on them if not none
    normalized_frames = []
    windowed_frames = []
    denoised_audios = []
    for i in tqdm(range(len(df_final)), desc="Processing"):
        y = df_final['y'].iloc[i]
        sr = df_final['sr'].iloc[i]
        frame_length = int(0.025 * sr)  # 25 ms frames
        hop_length = int(0.010 * sr)  # 10 ms hop length
        if y is not None and sr is not None and len(y) >= frame_length:
            norm_frames, win_frames, denoised_audio = preprocess_audio(y, sr, frame_length, hop_length)
            normalized_frames.append(norm_frames)
            windowed_frames.append(win_frames)
            denoised_audios.append(denoised_audio)
        else:
            normalized_frames.append(None)
            windowed_frames.append(None)
            denoised_audios.append(None)
    valid_indices = [i for i, (nf, wf, da) in enumerate(zip(normalized_frames, windowed_frames, denoised_audio))
                 if nf is not None and wf is not None and da is not None]

    # Filter everything
    df_filtered = df_final.iloc[valid_indices].reset_index(drop=True)
    normalized_frames = [normalized_frames[i] for i in valid_indices]
    windowed_frames = [windowed_frames[i] for i in valid_indices]
    denoised_audios = [denoised_audios[i] for i in valid_indices]

    #update the df_filtered with the new columns
    df_filtered['normalized_frames'] = normalized_frames
    df_filtered['windowed_frames'] = windowed_frames
    df_filtered['denoised_audios'] = denoised_audios

    print("removed counts from vad" , removed_counts)
    return df_filtered


def extract_features(df_filtered):
    output_file = 'b7_fifties.csv'
    header_written = False

    with open(output_file, mode='w', newline='') as file:
        writer = None

        for i in tqdm(range(len(df_filtered)), desc="Extracting features"):
            y = df_filtered['denoised_audios'].iloc[i]
            sr = df_filtered['sr'].iloc[i]

            if y is not None and sr is not None:
                features = extract_features_from_audio(
                    y, sr,
                    df_filtered.iloc[i],
                    df_filtered["windowed_frames"].iloc[i]
                )
                features['gender'] = df_filtered['gender'].iloc[i]
                features['age'] = df_filtered['age'].iloc[i]

                if not header_written:
                    writer = csv.DictWriter(file, fieldnames=features.keys())
                    writer.writeheader()
                    header_written = True

                writer.writerow(features)

    print(f"Saved features to {output_file}")

def calculate_features(y, sr, win_frames):
    features = {}

    # Duration
    duration = calculate_duration(y, sr)
    features['duration'] = duration

    # F0
    f0_features = calculate_f0_features(y, sr)
    features['f0_mean'] = f0_features['mean']
    features['f0_std'] = f0_features['std']
    features['f0_5_percentile'] = f0_features['5_percentile']
    features['f0_95_percentile'] = f0_features['95_percentile']

    # Tempo
    features['tempo'] = calculate_tempo(y, sr, duration)

    # Formants
    f1, f2, f3 = calculate_formants(y, sr)
    features['formant1'] = f1
    features['formant2'] = f2
    features['formant3'] = f3

    # MFCCs
    mfcc_features = calculate_mfcc(y, sr)

    for i, val in enumerate(mfcc_features['mfcc_means']):
        features[f'mfcc_{i}_mean'] = val
    for i, val in enumerate(mfcc_features['mfcc_stds']):
        features[f'mfcc_{i}_std'] = val

    for i, val in enumerate(mfcc_features['delta_means']):
        features[f'delta_mfcc_{i}_mean'] = val
    for i, val in enumerate(mfcc_features['delta_stds']):
        features[f'delta_mfcc_{i}_std'] = val

    for i, val in enumerate(mfcc_features['delta2_means']):
        features[f'delta2_mfcc_{i}_mean'] = val
    for i, val in enumerate(mfcc_features['delta2_stds']):
        features[f'delta2_mfcc_{i}_std'] = val
    for i, val in enumerate(mfcc_features['mfcc_kurtosis']):
        features[f'mfcc_{i}_kurtosis'] = val
    for i, val in enumerate(mfcc_features['mfcc_skewness']):
        features[f'mfcc_{i}_skewness'] = val

    # Cepstrum
    cpp_mean = compute_cpp_from_windowed(win_frames, sr)
    features['cpp_mean'] = cpp_mean

    # Short-Time Energy
    ste_features = short_time_energy(y, sr, win_frames)
    features['ste_mean'] = ste_features['mean_energy']
    features['ste_std'] = ste_features['std_energy']
    features['ste_max'] = ste_features['max_energy']
    features['ste_min'] = ste_features['min_energy']
    features['ste_variance'] = ste_features['energy_variance']
    
    return features

def extract_features_per_audio(path):
    y, sr = librosa.load(path, sr=None)
    frame_length = int(0.025 * sr)  # 25 ms frames
    hop_length = int(0.010 * sr)  # 10 ms hop length
    norm_frames, win_frames, denoised_audio = preprocess_audio(y, sr, frame_length, hop_length)
    features = calculate_features(denoised_audio, sr,win_frames)
    return pd.DataFrame([features])
# main
def main():
    # Define file paths
    file_path = "D:/3rd/2/NN/filtered_data_labeled.tsv"
    audio_dir = "D:/3rd/2/NN/audio_batch_7"

    # Step 1: Read data
    df = read_data(file_path, audio_dir)
    # Step 2: Load audio files
    df = load_audios(df)
    # Step 3: Filter data
    df_filtered = filter_data(df)
    # Step 4: Preprocess audio
    df_filtered = preprocess_audio_batch(df_filtered)
    # Step 5: Extract features
    features_df = extract_features(df_filtered)
    print("Features extraction completed.")
    # print(df_filtered.columns)
    # from_one_audio = extract_features_per_audio("D:/3rd/2/NN/oneAudio/common_voice_en_1459.mp3")
    # from_one_audio.to_csv("extracted_features_onne.csv", index=False)
if __name__ == "__main__":
    main()


















