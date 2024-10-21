import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class StateSpaceSimulation:
    def __init__(self, state_dim, A, B, C, D, sampling_interval, sim_duration=10):
        """
        Initialize the state-space simulation with given parameters.

        Args:
            state_dim (int): Number of states.
            A (np.ndarray): State transition matrix.
            B (np.ndarray): Input matrix.
            C (np.ndarray): Output matrix.
            D (np.ndarray): Feedforward matrix.
            sampling_interval (float): Sampling interval for discrete-time simulation.
            sim_duration (float): Total simulation duration in seconds.
        """
        self.state_dim = state_dim
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.sampling_interval = sampling_interval
        self.num_samples = int(sim_duration / sampling_interval)
        self.time_vector = np.linspace(0, sim_duration, self.num_samples)
    
    def generate_pwm(self, freq, duty_cycle, duration=5):
        """
        Generate a PWM signal with specified frequency, duty cycle, and duration.

        Args:
            freq (float): Frequency of the PWM signal in Hz.
            duty_cycle (float): Duty cycle as a percentage (0 to 100).
            duration (float): Duration of the signal in seconds.

        Returns:
            np.ndarray: PWM signal array.
        """
        time_points = np.linspace(0, duration, int(duration / self.sampling_interval))
        pwm_signal = np.zeros_like(time_points)
        period = 1 / freq
        high_time = period * (duty_cycle / 100)

        for i, t in enumerate(time_points):
            if t % period < high_time:
                pwm_signal[i] = 1
            else:
                pwm_signal[i] = 0

        return pwm_signal

    def generate_swept_sine(self, start_freq, end_freq, duration):
        """
        Generate an exponential swept-sine signal.

        Args:
            start_freq (float): Starting frequency in Hz.
            end_freq (float): Ending frequency in Hz.
            duration (float): Duration in seconds.

        Returns:
            np.ndarray: Swept-sine signal array.
        """
        sweep_rate = duration / np.log(end_freq / start_freq)
        swept_sine_signal = np.sin(2 * np.pi * start_freq * sweep_rate * (np.exp(self.time_vector / sweep_rate) - 1))

        return swept_sine_signal

    def simulate(self, input_signal):
        """
        Simulate the discrete-time state-space system with the given input signal.

        Args:
            input_signal (np.ndarray): Input signal array.

        Returns:
            np.ndarray: Output signal array.
        """
        num_steps = len(input_signal)
        state_vector = np.zeros(self.state_dim)
        output_signal = np.zeros(num_steps)

        for t in range(num_steps):
            current_input = np.atleast_1d(input_signal[t])
            output_signal[t] = self.C @ state_vector + self.D @ current_input
            state_vector = self.A @ state_vector + self.B @ current_input

        return output_signal

    def save_to_csv(self, output_signal, filename='output_signal.csv'):
        """
        Save the output signal to a CSV file.

        Args:
            output_signal (np.ndarray): Output signal array.
            filename (str): CSV file name.
        """
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, filename)
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Time [s]', 'Output Signal'])
            for t, y in zip(self.time_vector, output_signal):
                writer.writerow([t, y])

        print(f"Output signal saved as {file_path}")

    def plot_signals(self, input_signal, output_signal, filename='signal_comparison.png'):
        """
        Plot and save input and output signals for comparison.

        Args:
            input_signal (np.ndarray): Input signal array.
            output_signal (np.ndarray): Output signal array.
            filename (str): File name for the plot.
        """
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_vector, input_signal, label='Input Signal', linewidth=2)
        plt.plot(self.time_vector, output_signal, label='Output Signal', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Input vs Output Signal')
        plt.legend()
        plt.grid(True)

        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        plt.close()

        print(f"Plot saved as {file_path}")

    def compare_signals(self, original_signal, processed_signal, filename='comparison_plot.png'):
        """
        Plot and save the comparison between original and processed signals.

        Args:
            original_signal (np.ndarray): Original signal array.
            processed_signal (np.ndarray): Processed signal array.
            filename (str): File name for the plot.
        """
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.time_vector, original_signal, label='Original Signal', linewidth=2)
        plt.plot(self.time_vector, processed_signal, label='Processed Signal', linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Original vs Processed Signal')
        plt.legend()
        plt.grid(True)

        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        plt.close()

        print(f"Comparison plot saved as {file_path}")

    def run_simulation(self, signal_type='pwm', freq=1, duty_cycle=50, start_freq=0.1, end_freq=10):
        """
        Run the simulation with PWM or swept-sine input and save the results.

        Args:
            signal_type (str): Type of input signal ('pwm' or 'swept_sine').
            freq (float): Frequency for PWM signal.
            duty_cycle (float): Duty cycle for PWM signal.
            start_freq (float): Start frequency for swept-sine.
            end_freq (float): End frequency for swept-sine.
        """
        if signal_type == 'pwm':
            input_signal = self.generate_pwm(freq, duty_cycle)
        else:
            input_signal = self.generate_swept_sine(start_freq, end_freq, duration=10)

        output_signal = self.simulate(input_signal)

        self.save_to_csv(output_signal)
        self.plot_signals(input_signal, output_signal)
