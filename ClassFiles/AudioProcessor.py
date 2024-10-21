import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import soundfile as sf
from scipy.io import wavfile

class AudioProcessor:
    def __init__(self, control_period):
        """
        Initialize the AudioProcessor class with a control period.

        Args:
            control_period (float): Control period in seconds.
        """
        self.file_types = [('Audio Files', '*.wav *.mp3 *.m4a'), ('CSV Files', '*.csv')]
        self.control_period = control_period

    def load_csv_data(self):
        """
        Open a file dialog to select a CSV file and return its data column.

        Returns:
            np.ndarray: Data column from the selected CSV file.
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])

        if file_path:
            print(f"User selected: {file_path}")

            # Determine the number of columns in the CSV file
            with open(file_path, 'r') as file:
                first_line = file.readline().strip().split(',')
                num_columns = len(first_line)

            # Load data based on the number of columns
            if num_columns > 1:
                data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=1)
            else:
                data = np.loadtxt(file_path, delimiter=',')

            return data
        else:
            print("User canceled the file selection.")
            return None
    
    def load_audio_file(self, duration=10):
        """
        Load audio data from a file and return the time series data and sample rate.

        Args:
            duration (float): Duration of the audio to load in seconds.

        Returns:
            np.ndarray, int: Audio data and sampling rate.
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=self.file_types)

        if file_path:
            print(f"User selected: {file_path}")

            # Read the file and get the full sampling rate and data
            data, fs = sf.read(file_path, dtype='float32')
            
            # Calculate the number of samples to read based on the specified duration
            num_samples = int(duration * fs)
            
            # Trim the data to the specified duration
            data = data[:num_samples]
            
            return data, fs
        else:
            print("User canceled the file selection.")
            return None, None

    def convert_to_44_1kHz(self, control_data):
        """
        Convert control period data to 44.1kHz sampling rate using zero-order hold.

        Args:
            control_data (np.ndarray): Time series data sampled at the control period rate.

        Returns:
            np.ndarray: Time series data resampled at 44.1kHz.
        """
        control_fs = 1 / self.control_period
        target_fs = 44100

        repeat_factor = int(target_fs / control_fs)
        resampled_data = np.repeat(control_data, repeat_factor)

        return resampled_data

    def convert_to_control_period(self, audio_data, audio_fs):
        """
        Convert audio data sampled at a high rate to control period sampling.

        Args:
            audio_data (np.ndarray): Time series data of the audio.
            audio_fs (int): Original sampling frequency of the audio.

        Returns:
            np.ndarray: Time series data sampled at the control period rate.
        """
        control_fs = 1 / self.control_period
        step = int(audio_fs / control_fs)

        downsampled_data = audio_data[::step]

        return downsampled_data

    def save_as_csv(self, time_series_data, filename='output_data.csv'):
        """
        Save time series data as a CSV file.

        Args:
            time_series_data (np.ndarray): Time series data to save.
            filename (str): Name of the CSV file to save.
        """
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, filename)
        np.savetxt(file_path, time_series_data, delimiter=',')

        print(f"Data saved as {file_path}")

    def save_as_wav(self, time_series_data, sample_rate, filename='output_audio.wav'):
        """
        Save time series data as a WAV file.

        Args:
            time_series_data (np.ndarray): Time series data to save.
            sample_rate (int): Sampling rate for the WAV file.
            filename (str): Name of the WAV file to save.
        """
        # Check if the signal exceeds the range of -1 to 1
        if np.max(np.abs(time_series_data)) > 1:
            # If the signal exceeds the range, scale it to be within -1 to 1
            max_val = np.max(np.abs(time_series_data))
            time_series_data = time_series_data / max_val

        output_dir = './output'
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, filename)
        # Save the time series data as a WAV file
        wavfile.write(file_path, sample_rate, time_series_data.astype(np.float32))

        print(f"Audio saved as {file_path}")
    
    def process_csv_to_wav(self):
        """
        Select a CSV file, convert its data to 44.1kHz, and save as a WAV file.
        """
        csv_data = self.load_csv_data()
        if csv_data is not None:
            wav_data = self.convert_to_44_1kHz(csv_data * 4e-1)
            self.save_as_wav(wav_data, 44100, 'converted_from_csv.wav')
    
    def process_audio_to_csv(self):
        """
        Select an audio file, convert its data to control period rate, and save as a CSV file.
        """
        audio_data, audio_fs = self.load_audio_file()
        if audio_data is not None:
            control_data = self.convert_to_control_period(audio_data, audio_fs)
            self.save_as_csv(control_data / 4e-1, 'converted_from_audio.csv')
    