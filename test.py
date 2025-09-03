import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch, correlate
import os
import warnings
warnings.filterwarnings('ignore')

class PulseEchoProcessor:
    def __init__(self, sampling_rate=12500000, target_rate=125000, echo_window=1.0):
        """
        Initialize the Pulse and Echo Processor
        
        Parameters:
        sampling_rate (int): Original sampling rate in Hz (default: 12.5 MS/s)
        target_rate (int): Target sampling rate after downsampling (default: 125 kS/s)
        echo_window (float): Time window after each pulse to capture echoes (default: 1.0 seconds)
        """
        self.original_sampling_rate = sampling_rate
        self.target_sampling_rate = target_rate
        self.downsample_factor = sampling_rate // target_rate
        self.echo_window = echo_window  # Time window for echoes after each pulse
        
        # Storage for results
        self.pulses = []  # To store detected pulses with echo windows
        self.averaged_pulse_echo = None  # To store the averaged pulse with echoes
        self.echo_segments = []  # To store just the echo portions
        self.averaged_echo = None  # To store the averaged echo
        
    def load_data(self, filepath, time_col=0, signal_col=1):
        """
        Load data from CSV file
        
        Parameters:
        filepath (str): Path to the CSV file
        time_col (int): Column index for time data
        signal_col (int): Column index for signal data
        
        Returns:
        tuple: (time_array, signal_array) or (None, None) if error
        """
        try:
            print(f"Loading data from {filepath}...")
            data = pd.read_csv(filepath)
            
            # Extract time and signal data
            time = data.iloc[:, time_col].values
            signal = data.iloc[:, signal_col].values
            
            print(f"Original data: {len(signal)} samples at {self.original_sampling_rate/1e6:.1f} MS/s")
            
            return time, signal
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
            
    def downsample(self, time, signal):
        """
        Downsample the signal to the target rate
        
        Parameters:
        time (array): Original time array
        signal (array): Original signal array
        
        Returns:
        tuple: (downsampled_time, downsampled_signal)
        """
        print(f"Downsampling from {self.original_sampling_rate/1e6:.1f} MS/s to {self.target_sampling_rate/1e3:.1f} kS/s")
        
        # Calculate the number of samples after downsampling
        n_samples = len(signal) // self.downsample_factor
        
        # Downsample by selecting every nth sample
        downsampled_signal = signal[:n_samples * self.downsample_factor:self.downsample_factor]
        downsampled_time = time[:n_samples * self.downsample_factor:self.downsample_factor]
        
        print(f"Downsampled data: {len(downsampled_signal)} samples")
        
        return downsampled_time, downsampled_signal
        
    def apply_filter(self, signal, lowcut=1000, highcut=20000, order=4):
        """
        Apply a bandpass filter to the signal
        
        Parameters:
        signal (array): Input signal
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        order (int): Filter order
        
        Returns:
        array: Filtered signal
        """
        # Calculate Nyquist frequency
        nyquist = 0.5 * self.target_sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design bandpass filter
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = filtfilt(b, a, signal)
        
        print(f"Applied bandpass filter: {lowcut/1000:.1f} - {highcut/1000:.1f} kHz")
        
        return filtered_signal
        
    def detect_pulses(self, time, signal, min_height=0.3, min_distance=0.001, n_pulses=10):
        """
        Detect pulses in the signal
        
        Parameters:
        time (array): Time array
        signal (array): Signal array
        min_height (float): Minimum height for pulse detection (relative to max)
        min_distance (float): Minimum distance between pulses in seconds
        n_pulses (int): Number of pulses to detect
        
        Returns:
        list: List of pulse indices
        """
        # Normalize signal for detection
        abs_signal = np.abs(signal)
        norm_signal = abs_signal / np.max(abs_signal)
        
        # Convert min_distance from seconds to samples
        min_distance_samples = int(min_distance * self.target_sampling_rate)
        
        # Find peaks (pulses)
        peaks, properties = find_peaks(
            norm_signal, 
            height=min_height,
            distance=min_distance_samples,
            prominence=0.2
        )
        
        # If we found more peaks than needed, select the strongest ones
        if len(peaks) > n_pulses:
            # Sort by prominence (strength of the peak)
            prominences = properties['prominences']
            strongest_peaks = peaks[np.argsort(prominences)[-n_pulses:]]
            peaks = np.sort(strongest_peaks)
        
        print(f"Detected {len(peaks)} pulses at positions: {peaks}")
        
        return peaks
        
    def extract_pulse_echo_segments(self, time, signal, pulse_indices):
        """
        Extract pulse segments with echo windows
        
        Parameters:
        time (array): Time array
        signal (array): Signal array
        pulse_indices (list): List of pulse indices
        
        Returns:
        tuple: (pulse_echo_segments, echo_only_segments)
        """
        # Convert echo window from seconds to samples
        echo_window_samples = int(self.echo_window * self.target_sampling_rate)
        
        pulse_echo_segments = []
        echo_only_segments = []
        
        for idx in pulse_indices:
            # Define window around the pulse (pulse + echo window)
            start_idx = idx
            end_idx = min(len(signal), idx + echo_window_samples)
            
            # Extract the pulse with echo window
            segment = signal[start_idx:end_idx]
            
            # If the segment is shorter than expected, pad with zeros
            if len(segment) < echo_window_samples:
                padding = np.zeros(echo_window_samples - len(segment))
                segment = np.concatenate([segment, padding])
                
            pulse_echo_segments.append(segment)
            
            # Extract echo-only portion (skip the direct pulse)
            # Assuming the direct pulse lasts about 0.1ms, skip first 0.2ms
            skip_samples = int(0.0002 * self.target_sampling_rate)
            echo_only = segment[skip_samples:]
            echo_only_segments.append(echo_only)
            
        return pulse_echo_segments, echo_only_segments
        
    def align_segments(self, segments, reference_index=0):
        """
        Align segments using cross-correlation
        
        Parameters:
        segments (list): List of segments to align
        reference_index (int): Index of reference segment
        
        Returns:
        list: List of aligned segments
        """
        if not segments:
            return segments
            
        # Use the first segment as reference
        reference = segments[reference_index]
        aligned_segments = [reference]
        
        for i in range(len(segments)):
            if i == reference_index:
                continue
                
            # Cross-correlate with reference
            correlation = np.correlate(segments[i], reference, mode='full')
            
            # Find the lag that maximizes correlation
            lag = np.argmax(correlation) - (len(reference) - 1)
            
            # Shift the segment
            if lag > 0:
                aligned_segment = np.concatenate([np.zeros(lag), segments[i][:-lag]])
            elif lag < 0:
                aligned_segment = np.concatenate([segments[i][-lag:], np.zeros(-lag)])
            else:
                aligned_segment = segments[i]
                
            # Ensure the aligned segment has the same length as reference
            if len(aligned_segment) > len(reference):
                aligned_segment = aligned_segment[:len(reference)]
            elif len(aligned_segment) < len(reference):
                aligned_segment = np.concatenate([aligned_segment, np.zeros(len(reference) - len(aligned_segment))])
                
            aligned_segments.append(aligned_segment)
            
        return aligned_segments
        
    def average_segments(self, segments):
        """
        Average multiple segments
        
        Parameters:
        segments (list): List of segments
        
        Returns:
        array: Averaged segment
        """
        if not segments:
            return None
            
        # Ensure all segments have the same length
        min_length = min(len(segment) for segment in segments)
        trimmed_segments = [segment[:min_length] for segment in segments]
        
        # Convert to numpy array for efficient computation
        segment_matrix = np.array(trimmed_segments)
        
        # Average the segments
        averaged_segment = np.mean(segment_matrix, axis=0)
        
        print(f"Averaged {len(segments)} segments")
        
        return averaged_segment
        
    def process_file(self, filepath, n_pulses=10):
        """
        Complete processing of a file: load, downsample, detect pulses, extract echoes, average
        
        Parameters:
        filepath (str): Path to the CSV file
        n_pulses (int): Number of pulses to detect and average
        
        Returns:
        tuple: (time, signal, pulse_echo_segments, echo_only_segments, averaged_pulse_echo, averaged_echo) or None if error
        """
        # Load data
        time, signal = self.load_data(filepath)
        if time is None:
            return None
            
        # Downsample
        downsampled_time, downsampled_signal = self.downsample(time, signal)
        
        # Apply filter
        filtered_signal = self.apply_filter(downsampled_signal)
        
        # Detect pulses
        pulse_indices = self.detect_pulses(downsampled_time, filtered_signal, n_pulses=n_pulses)
        
        # Extract pulse segments with echo windows
        pulse_echo_segments, echo_only_segments = self.extract_pulse_echo_segments(
            downsampled_time, filtered_signal, pulse_indices)
        
        # Align segments
        aligned_pulse_echo = self.align_segments(pulse_echo_segments)
        aligned_echo_only = self.align_segments(echo_only_segments)
        
        # Average segments
        averaged_pulse_echo = self.average_segments(aligned_pulse_echo)
        averaged_echo = self.average_segments(aligned_echo_only)
        
        # Store results
        self.downsampled_time = downsampled_time
        self.filtered_signal = filtered_signal
        self.pulse_indices = pulse_indices
        self.pulse_echo_segments = aligned_pulse_echo
        self.echo_only_segments = aligned_echo_only
        self.averaged_pulse_echo = averaged_pulse_echo
        self.averaged_echo = averaged_echo
        
        return (downsampled_time, filtered_signal, aligned_pulse_echo, 
                aligned_echo_only, averaged_pulse_echo, averaged_echo)
        
    def plot_results(self, save_path=None):
        """
        Plot the processing results
        
        Parameters:
        save_path (str): Path to save the plot (optional)
        """
        if not hasattr(self, 'filtered_signal') or not self.pulse_echo_segments:
            print("No data to plot. Run process_file() first.")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Full signal with detected pulses
        axes[0, 0].plot(self.downsampled_time, self.filtered_signal, 'b-', label='Filtered Signal', alpha=0.7)
        axes[0, 0].plot(self.downsampled_time[self.pulse_indices], 
                       self.filtered_signal[self.pulse_indices], 
                       'ro', label='Detected Pulses', markersize=8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Signal with Detected Pulses')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Individual pulse+echo segments
        segment_time = np.arange(len(self.pulse_echo_segments[0])) / self.target_sampling_rate * 1000  # Time in ms
        for i, segment in enumerate(self.pulse_echo_segments):
            axes[0, 1].plot(segment_time, segment, alpha=0.7, label=f'Segment {i+1}')
        axes[0, 1].set_xlabel('Time after pulse (ms)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].set_title('Individual Pulse+Echo Segments')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Averaged pulse+echo
        axes[1, 0].plot(segment_time, self.averaged_pulse_echo, 'r-', linewidth=2, label='Averaged Pulse+Echo')
        axes[1, 0].set_xlabel('Time after pulse (ms)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].set_title('Averaged Pulse with Echoes')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Echo-only segments and average
        echo_time = np.arange(len(self.echo_only_segments[0])) / self.target_sampling_rate * 1000  # Time in ms
        for i, echo in enumerate(self.echo_only_segments):
            axes[1, 1].plot(echo_time, echo, 'b-', alpha=0.3)
        axes[1, 1].plot(echo_time, self.averaged_echo, 'r-', linewidth=3, label='Averaged Echo')
        axes[1, 1].set_xlabel('Time after pulse (ms)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].set_title('Echo Segments with Average')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def detect_echoes_in_average(self, min_height=0.05, min_distance=0.0001):
        """
        Detect individual echoes in the averaged echo segment
        
        Parameters:
        min_height (float): Minimum height for echo detection (relative to max)
        min_distance (float): Minimum distance between echoes in seconds
        
        Returns:
        tuple: (echo_indices, echo_times, echo_amplitudes)
        """
        if self.averaged_echo is None:
            print("No averaged echo available. Run process_file() first.")
            return None, None, None
            
        # Normalize echo for detection
        abs_echo = np.abs(self.averaged_echo)
        norm_echo = abs_echo / np.max(abs_echo)
        
        # Convert min_distance from seconds to samples
        min_distance_samples = int(min_distance * self.target_sampling_rate)
        
        # Find peaks (echoes)
        echo_indices, properties = find_peaks(
            norm_echo, 
            height=min_height,
            distance=min_distance_samples,
            prominence=0.1
        )
        
        # Convert indices to times (ms)
        echo_times = echo_indices / self.target_sampling_rate * 1000
        
        # Get echo amplitudes
        echo_amplitudes = self.averaged_echo[echo_indices]
        
        print(f"Detected {len(echo_indices)} echoes in averaged signal")
        
        return echo_indices, echo_times, echo_amplitudes
        
    def save_results(self, output_dir):
        """
        Save the processed results to files
        
        Parameters:
        output_dir (str): Directory to save results
        """
        if not hasattr(self, 'averaged_pulse_echo') or not self.pulse_echo_segments:
            print("No results to save. Run process_file() first.")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save averaged pulse+echo
        segment_time = np.arange(len(self.averaged_pulse_echo)) / self.target_sampling_rate * 1000  # Time in ms
        averaged_data = np.column_stack((segment_time, self.averaged_pulse_echo))
        np.savetxt(os.path.join(output_dir, "averaged_pulse_echo.csv"), averaged_data, 
                  delimiter=",", header="Time (ms),Amplitude", comments="")
        
        # Save averaged echo only
        echo_time = np.arange(len(self.averaged_echo)) / self.target_sampling_rate * 1000  # Time in ms
        echo_data = np.column_stack((echo_time, self.averaged_echo))
        np.savetxt(os.path.join(output_dir, "averaged_echo.csv"), echo_data,
                  delimiter=",", header="Time (ms),Amplitude", comments="")
        
        # Save individual pulse+echo segments
        for i, segment in enumerate(self.pulse_echo_segments):
            segment_data = np.column_stack((segment_time, segment))
            np.savetxt(os.path.join(output_dir, f"pulse_echo_segment_{i+1}.csv"), segment_data,
                      delimiter=",", header="Time (ms),Amplitude", comments="")
        
        # Save processing parameters
        with open(os.path.join(output_dir, "processing_parameters.txt"), "w") as f:
            f.write("Pulse and Echo Processing Parameters\n")
            f.write("=" * 40 + "\n")
            f.write(f"Original sampling rate: {self.original_sampling_rate} Hz\n")
            f.write(f"Target sampling rate: {self.target_sampling_rate} Hz\n")
            f.write(f"Downsample factor: {self.downsample_factor}\n")
            f.write(f"Echo window: {self.echo_window} seconds\n")
            f.write(f"Number of pulses detected: {len(self.pulse_echo_segments)}\n")
            f.write(f"Pulse+echo segment length: {len(self.averaged_pulse_echo)} samples\n")
            f.write(f"Echo-only segment length: {len(self.averaged_echo)} samples\n")
        
        # Detect and save echo information
        echo_indices, echo_times, echo_amplitudes = self.detect_echoes_in_average()
        if echo_indices is not None:
            echo_info = np.column_stack((echo_times, echo_amplitudes))
            np.savetxt(os.path.join(output_dir, "detected_echoes.csv"), echo_info,
                      delimiter=",", header="Time (ms),Amplitude", comments="")
        
        print(f"Results saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Create processor instance
    processor = PulseEchoProcessor(
        sampling_rate=50000000, 
        target_rate=500000, 
        echo_window=1.0  # 1 second echo window
    )
    
    # Process a file
    filepath = "C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\Before avg\\Baseline.csv"  # Update with your file path
    results = processor.process_file(filepath, n_pulses=10)
    
    if results is not None:
        # Plot results
        processor.plot_results(save_path="C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\enhanced_analysis")

        # Detect echoes in the averaged signal
        echo_indices, echo_times, echo_amplitudes = processor.detect_echoes_in_average()
        if echo_indices is not None:
            print(f"Detected {len(echo_indices)} echoes:")
            for i, (t, a) in enumerate(zip(echo_times, echo_amplitudes)):
                print(f"Echo {i+1}: {t:.2f} ms, amplitude: {a:.6f}")
        
        # Save results
        processor.save_results("C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\enhanced_analysis")

        # Access the results
        (downsampled_time, filtered_signal, pulse_echo_segments, 
         echo_only_segments, averaged_pulse_echo, averaged_echo) = results
        print(f"Processing complete. Processed {len(pulse_echo_segments)} pulse+echo segments.")
    else:
        print("Processing failed.")