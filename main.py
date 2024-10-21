import numpy as np
from ClassFiles.AudioProcessor import AudioProcessor
from ClassFiles.StateSpaceSimulation import StateSpaceSimulation

def main():
    """
    Main function for selecting and executing different processing tasks.
    """
    # Display processing options to the user
    print("Select the processing option:")
    print("1. Simulate system and convert to CSV, then to WAV")
    print("2. Load CSV file and convert to WAV")
    print("3. Load audio file and convert to CSV")
    print("4. Compare two CSV files")

    # Get user input
    choice = input("Enter your choice (1-4): ")

    # Control period for simulation and conversion (e.g., 0.01s for 100Hz)
    control_period = 1e-2
    audio_processor = AudioProcessor(control_period)

    if choice == '1':
        # Processing 1: Simulation, save to CSV, then convert to WAV
        print("Starting system simulation...")
        
        # Define simulation parameters
        sampling_time = 1e-2
        system_order = 2
        A = np.array([[1, 1], [-0.5, -0.5]])
        B = np.array([[0], [1]])
        C = np.array([1, 0])
        D = np.array([0])
        
        # Initialize the simulation
        simulation = StateSpaceSimulation(system_order, A, B, C, D, sampling_time)
        
        # Generate PWM input signal
        pwm_signal = simulation.generate_pwm(freq=1, duty_cycle=50, duration=10)
        
        # Simulate system response
        output_signal = simulation.simulate(pwm_signal)
        
        # Save the simulation result as a CSV file
        simulation.save_to_csv(output_signal, 'simulation_output.csv')
        print("Simulation output saved as 'simulation_output.csv'.")

        # Convert CSV to WAV
        print("Converting simulation output to WAV...")
        audio_processor.process_csv_to_wav()
        print("Conversion completed: 'converted_from_csv.wav'.")

    elif choice == '2':
        # Processing 2: Load CSV file, convert to WAV
        print("Select a CSV file to convert to WAV...")
        audio_processor.process_csv_to_wav()
        print("CSV to WAV conversion completed.")

    elif choice == '3':
        # Processing 3: Load audio file, convert to CSV
        print("Select an audio file to convert to CSV...")
        audio_processor.process_audio_to_csv()
        print("Audio to CSV conversion completed.")

    elif choice == '4':
        # Processing 4: Load two CSV files and compare
        print("Select two CSV files for comparison...")

        # Load the first CSV file
        csv_data_1 = audio_processor.load_csv_data()
        if csv_data_1 is None:
            print("No file selected for the first CSV. Exiting comparison.")
            return

        # Load the second CSV file
        csv_data_2 = audio_processor.load_csv_data()
        if csv_data_2 is None:
            print("No file selected for the second CSV. Exiting comparison.")
            return

        # Compare the two CSV files
        simulation = StateSpaceSimulation(0, None, None, None, None, control_period)
        simulation.compare_signals(csv_data_1, csv_data_2, 'csv_comparison_plot.png')
        print("Comparison completed and saved as 'csv_comparison_plot.png'.")

    else:
        print("Invalid choice. Please select a valid option (1-4).")

if __name__ == "__main__":
    main()
