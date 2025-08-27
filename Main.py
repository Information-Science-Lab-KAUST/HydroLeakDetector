import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, correlate, stft, welch
import glob
import os
import warnings
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
warnings.filterwarnings('ignore')


from Vierwer import PipePlot as p_plot
from UI.ui import Ui_Form

class EnhancedLeakDetector(p_plot):
    def __init__(self, sampling_rate=12500000, pipe_length=1.0):
        self.sampling_rate = sampling_rate
        self.pipe_length = pipe_length
        self.dt = 1.0 / sampling_rate
        self.downsample_factor = 100
        self.effective_sampling_rate = sampling_rate / self.downsample_factor
        self.effective_dt = 1.0 / self.effective_sampling_rate
        self.speed_of_sound = 1480
        


    def starter(self):
        app = QtWidgets.QApplication(sys.argv)
        self.Main_div = QtWidgets.QWidget()
        self.Main_ui = Ui_Form()
        self.Main_ui.setupUi(self.Main_div)
        self.Main_div.show()
        self.clicked()
        sys.exit(app.exec_())

    def clicked(self):
        self.Main_ui.Baseline_btn.clicked.connect(self.get_baseline_path)
        self.Main_ui.Response_btn.clicked.connect(self.get_response_path)
        self.Main_ui.Plot_btn.clicked.connect(self.run_analysis)


    def get_baseline_path(self):

        file_path, _ = QFileDialog.getOpenFileName(
        None,                     
        "Select Baseline File",   
        "",                       
        "CSV Files (*.csv);;All Files (*)"  
    )
        if file_path:  
            self.Main_ui.Baseline_le.setText(file_path)  
            print(f"Selected Baseline File: {file_path}")

        self.g_baseline_path = file_path


    def get_response_path(self):

        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select Response File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.Main_ui.Response_le.setText(file_path)
            print(f"Selected Response File: {file_path}")
        self.g_response_path = file_path


    def run_analysis(self):
        baseline_path = self.g_baseline_path
        response_path = self.g_response_path
        self.main(baseline_path, response_path)

    def load_and_preprocess(self, filepath, time_col=0, signal_col=1):
        try:
            print(f"Loading {filepath}...")
            data = pd.read_csv(filepath)
            if len(data.columns) <= max(time_col, signal_col):
                time_col, signal_col = 0, 1
            time = data.iloc[:, time_col].values
            signal = data.iloc[:, signal_col].values
            print(f"Original: {len(signal)} samples at {self.sampling_rate/1e6:.1f} MS/s")
            if len(signal) > 100000:
                signal = self.downsample(signal)
                time = time[::self.downsample_factor]
                print(f"Downsampled: {len(signal)} samples at {self.effective_sampling_rate/1e3:.1f} kS/s")
            return time, signal
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None
            
    def downsample(self, signal):
        return signal[::self.downsample_factor]
        
    def apply_adaptive_filter(self, signal, pinger_freq=10000):
        signal = signal - np.mean(signal)
        f, Pxx = welch(signal, self.effective_sampling_rate, nperseg=1024)
        noise_floor = np.percentile(Pxx, 10)
        signal_power = np.max(Pxx)
        if signal_power / noise_floor > 100:
            lowcut = max(1000, pinger_freq * 0.5)
            highcut = min(20000, pinger_freq * 2)
        else:
            lowcut = max(500, pinger_freq * 0.8)
            highcut = min(15000, pinger_freq * 1.2)
        print(f"Applying bandpass filter: {lowcut/1000:.1f} - {highcut/1000:.1f} kHz")
        nyquist = 0.5 * self.effective_sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        if low >= 1 or high >= 1:
            b, a = butter(4, high, btype='low')
        else:
            b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)
        
    def find_direct_pulse(self, signal, method='energy'):
        if method == 'energy':
            window_size = int(0.0001 / self.effective_dt)
            energy = np.convolve(signal**2, np.ones(window_size)/window_size, mode='same')
            pulse_start = np.argmax(energy)
            return pulse_start
        elif method == 'peak':
            return np.argmax(np.abs(signal))
        elif method == 'matched_filter':
            template = np.exp(-np.linspace(-3, 3, 100)**2) * np.sin(2 * np.pi * 10000 * np.linspace(0, 0.0001, 100))
            correlation = np.correlate(np.abs(signal), np.abs(template), mode='same')
            return np.argmax(correlation)
            
    def calibrate_with_end_cap(self, time, signal):
        direct_idx = self.find_direct_pulse(signal)
        expected_delay = (2 * self.pipe_length) / 1480
        search_start = direct_idx + int(0.8 * expected_delay / self.effective_dt)
        search_end = direct_idx + int(1.2 * expected_delay / self.effective_dt)
        search_end = min(search_end, len(signal) - 1)
        if search_end <= search_start:
            print("Warning: Cannot find end cap reflection, using default speed of sound")
            return 1480, direct_idx, direct_idx
        search_signal = np.abs(signal[search_start:search_end])
        end_cap_idx = search_start + np.argmax(search_signal)
        time_diff = time[end_cap_idx] - time[direct_idx]
        if time_diff <= 0:
            print("Warning: Invalid time difference, using default speed of sound")
            return 1480, direct_idx, end_cap_idx
        calculated_speed = (2 * self.pipe_length) / time_diff
        if not (1000 <= calculated_speed <= 2000):
            print(f"Warning: Unrealistic speed of sound {calculated_speed:.1f} m/s, using default")
            return 1480, direct_idx, end_cap_idx
        print(f"Calibrated speed of sound: {calculated_speed:.1f} m/s")
        return calculated_speed, direct_idx, end_cap_idx
        
    def align_signals(self, baseline_signal, scenario_signal):
        min_length = min(len(baseline_signal), len(scenario_signal))
        baseline_trunc = baseline_signal[:min_length]
        scenario_trunc = scenario_signal[:min_length]
        correlation = correlate(baseline_trunc, scenario_trunc, mode='full')
        lags = np.arange(-len(scenario_trunc) + 1, len(baseline_trunc))
        best_lag = lags[np.argmax(correlation)]
        return best_lag, min_length
        
    def detect_leak_echoes(self, baseline_signal, scenario_signal, time, direct_idx):
        min_length = min(len(baseline_signal), len(scenario_signal))
        baseline_signal = baseline_signal[:min_length]
        scenario_signal = scenario_signal[:min_length]
        time = time[:min_length]
        max_time = (2 * self.pipe_length) / self.speed_of_sound
        max_idx = direct_idx + int(max_time / self.effective_dt)
        max_idx = min(max_idx, len(time) - 1)
        search_start = direct_idx + int(0.0001 / self.effective_dt)
        search_end = max_idx
        diff_signal = np.abs(scenario_signal) - np.abs(baseline_signal)
        baseline_abs = np.abs(baseline_signal[search_start:search_end])
        scenario_abs = np.abs(scenario_signal[search_start:search_end])
        norm_diff = (scenario_abs - baseline_abs) / (baseline_abs + 1e-10)
        window_size = int(0.00005 / self.effective_dt)
        if window_size < 5:
            window_size = 5
        energy_baseline = np.convolve(baseline_abs**2, np.ones(window_size)/window_size, mode='same')
        energy_scenario = np.convolve(scenario_abs**2, np.ones(window_size)/window_size, mode='same')
        energy_diff = energy_scenario - energy_baseline
        combined_evidence = np.zeros_like(diff_signal[search_start:search_end])
        combined_evidence += diff_signal[search_start:search_end] / np.max(np.abs(diff_signal[search_start:search_end]))
        combined_evidence += norm_diff / np.max(np.abs(norm_diff))
        combined_evidence += energy_diff / np.max(np.abs(energy_diff))
        min_height = 0.5
        min_distance = int(0.0001 / self.effective_dt)
        peaks, properties = find_peaks(
            combined_evidence, 
            height=min_height,
            distance=min_distance,
            prominence=0.3
        )
        echo_indices = peaks + search_start
        echo_metrics = []
        for idx in echo_indices:
            if idx >= len(time):
                continue
            time_diff = time[idx] - time[direct_idx]
            distance = (self.speed_of_sound * time_diff) / 2
            if 0 < distance < self.pipe_length:
                echo_metrics.append({
                    'time': time[idx],
                    'time_diff': time_diff,
                    'distance': distance,
                    'amplitude': scenario_signal[idx],
                    'evidence': combined_evidence[idx - search_start]
                })
        echo_metrics.sort(key=lambda x: x['evidence'], reverse=True)
        primary_leak = echo_metrics[0] if echo_metrics else None
        return echo_indices, echo_metrics, primary_leak
        
    def plot_detailed_analysis(self, baseline_time, baseline_signal, scenario_signal,
                           direct_idx, echo_indices, echo_metrics, primary_leak,
                           scenario_name, save_path=None):

        min_length = min(len(baseline_time), len(scenario_signal))
        baseline_time = baseline_time[:min_length]
        baseline_signal = baseline_signal[:min_length]
        scenario_signal = scenario_signal[:min_length]
        echo_indices = [idx for idx in echo_indices if idx < min_length]

        # Clear previous plots in the frame
        for i in reversed(range(self.Main_ui.plot_frame.layout().count())):
            self.Main_ui.plot_frame.layout().itemAt(i).widget().setParent(None)

        # Create a figure and canvas
        fig = Figure(figsize=(12, 12))
        canvas = FigureCanvas(fig)
        self.Main_ui.plot_frame.layout().addWidget(canvas)  # Add canvas to your Qt frame
        axes = fig.subplots(3, 1)

        # --- Raw signals ---
        axes[0].plot(baseline_time, baseline_signal, 'b-', label='Baseline', alpha=0.7)
        axes[0].plot(baseline_time, scenario_signal, 'r-', label=scenario_name, alpha=0.7)
        axes[0].axvline(x=baseline_time[direct_idx], color='green', linestyle='--', label='Direct Pulse')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Raw Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # --- Difference signal ---
        diff_signal = np.abs(scenario_signal) - np.abs(baseline_signal)
        axes[1].plot(baseline_time, diff_signal, 'purple', label='Difference (Scenario - Baseline)', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

        for i, idx in enumerate(echo_indices):
            if idx < len(baseline_time):
                is_primary = False
                for echo in echo_metrics:
                    if abs(baseline_time[idx] - echo['time']) < 1e-6:
                        if echo.get('primary', False):
                            is_primary = True
                            break
                color = 'green' if is_primary else 'red'
                marker = '*' if is_primary else 'o'
                size = 12 if is_primary else 8
                axes[1].plot(baseline_time[idx], diff_signal[idx], marker, color=color, markersize=size)
                label = f'Primary Leak' if is_primary else f'Echo {i+1}'
                axes[1].annotate(label, xy=(baseline_time[idx], diff_signal[idx]),
                                xytext=(10, 10), textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude Difference')
        axes[1].set_title('Difference Signal (|Scenario| - |Baseline|)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # --- Leak distance estimates ---
        if echo_metrics:
            distances = [echo['distance'] for echo in echo_metrics]
            evidence = [echo['evidence'] for echo in echo_metrics]
            is_primary = [echo.get('primary', False) for echo in echo_metrics]
            colors = ['green' if primary else 'orange' for primary in is_primary]
            bars = axes[2].bar(range(len(distances)), distances, color=colors, alpha=0.7)
            axes[2].set_xlabel('Echo Number')
            axes[2].set_ylabel('Distance (m)')
            axes[2].set_title('Estimated Leak Distances (Green = Primary)')
            for i, (d, e, primary) in enumerate(zip(distances, evidence, is_primary)):
                text_color = 'black' if not primary else 'green'
                weight = 'bold' if primary else 'normal'
                axes[2].text(i, d, f'{d:.3f}m\n({e:.2f})', ha='center', va='bottom',
                            color=text_color, weight=weight)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        canvas.draw()  # Render the plot inside the Qt frame

        
    def analyze_scenario(self, baseline_path, scenario_path, output_dir):
        """Analyze a single scenario against baseline"""

        baseline_time, baseline_raw = self.load_and_preprocess(baseline_path)
        scenario_time, scenario_raw = self.load_and_preprocess(scenario_path)
        
        if baseline_time is None or scenario_time is None:
            print("Error: Could not load data files")
            return None, None, None
            

        baseline_filtered = self.apply_adaptive_filter(baseline_raw)
        scenario_filtered = self.apply_adaptive_filter(scenario_raw)
        

        min_length = min(len(baseline_filtered), len(scenario_filtered))
        baseline_filtered = baseline_filtered[:min_length]
        scenario_filtered = scenario_filtered[:min_length]
        baseline_time = baseline_time[:min_length]
        

        self.speed_of_sound, direct_idx, end_cap_idx = self.calibrate_with_end_cap(
            baseline_time, baseline_filtered)
        
  
        echo_indices, echo_metrics, primary_leak = self.detect_leak_echoes(
            baseline_filtered, scenario_filtered, baseline_time, direct_idx)
        
    
        if primary_leak:
            for echo in echo_metrics:
                echo['primary'] = (abs(echo['time'] - primary_leak['time']) < 1e-6)
        
     
        os.makedirs(output_dir, exist_ok=True)
        
       
        scenario_name = os.path.splitext(os.path.basename(scenario_path))[0]
        plot_path = os.path.join(output_dir, f"{scenario_name}_analysis.png")
        
        self.plot_detailed_analysis(
            baseline_time, baseline_filtered, scenario_filtered,
            direct_idx, echo_indices, echo_metrics, primary_leak, scenario_name, plot_path)
        
    
        print(f"\n=== RESULTS FOR {scenario_name} ===")
        
        if echo_metrics:
            print(f"Detected {len(echo_metrics)} potential echoes:")
            print("-" * 60)
            
            for i, echo in enumerate(echo_metrics):
                is_primary = echo.get('primary', False)
                marker = "★" if is_primary else "○"
                
                print(f"{marker} Echo {i+1}:")
                print(f"   Distance: {echo['distance']:.3f} m from sensor")
                print(f"   Time delay: {echo['time_diff']*1000:.2f} ms")
                print(f"   Evidence score: {echo['evidence']:.3f}")
                
                if is_primary:
                    print("   STATUS: PRIMARY LEAK CANDIDATE")
                
                print()
            
        
            if primary_leak:
                print("=== VALIDATION ===")
                print(f"Primary leak detected at: {primary_leak['distance']:.3f} m")
                print(f"Time delay: {primary_leak['time_diff']*1000:.2f} ms")
                print(f"Evidence score: {primary_leak['evidence']:.3f}")
                
          
                if 0 < primary_leak['distance'] < self.pipe_length:
                    print("✓ Distance is within pipe bounds (0 - 1.0 m)")
                else:
                    print("✗ Warning: Distance is outside pipe bounds")
                
       
                if primary_leak['evidence'] > 1.0:
                    print("✓ Strong evidence for leak detection")
                elif primary_leak['evidence'] > 0.5:
                    print("~ Moderate evidence for leak detection")
                else:
                    print("✗ Weak evidence for leak detection")
        else:
            print("No leak echoes detected")
            
        
            print("\n=== DIAGNOSTICS ===")
            print(f"Direct pulse found at: {baseline_time[direct_idx]:.6f} s")
            print(f"End cap reflection at: {baseline_time[end_cap_idx]:.6f} s")
            print(f"Speed of sound: {self.speed_of_sound:.1f} m/s")
            

            correlation = np.corrcoef(baseline_filtered, scenario_filtered)[0, 1]
            print(f"Correlation between signals: {correlation:.3f}")
            
            if correlation > 0.99:
                print("✗ Signals are very similar - possible issues with leak simulation")
            else:
                print("✓ Signals are different - leak should be detectable")
                
   
            baseline_energy = np.sum(baseline_filtered**2)
            scenario_energy = np.sum(scenario_filtered**2)
            energy_ratio = scenario_energy / baseline_energy
            
            print(f"Energy ratio (scenario/baseline): {energy_ratio:.3f}")
            
            if energy_ratio < 1.1:
                print("✗ Little additional energy in scenario - leak might be too weak")
            else:
                print("✓ Additional energy detected in scenario - leak should be present")
        
        return echo_metrics, baseline_time, scenario_filtered

    def main(self,baseline_path , scenario_path):

        detector = EnhancedLeakDetector(
            sampling_rate=50000000,  # 12.5 MS/s
            pipe_length=1.0          # 1 meter pipe
        )
        
        # Define paths
       
        output_dir = "C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\enhanced_analysis"


        scenario_files = glob.glob(f"{scenario_path}")

        scenario_files = [f for f in scenario_files if "baseline" not in f.lower()]
        
        print(f"Found {len(scenario_files)} scenario files")

        all_results = []
        
        for scenario_path in scenario_files:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(scenario_path)}")
            print(f"{'='*60}")
            
            results, time, signal = detector.analyze_scenario(
                baseline_path, scenario_path, output_dir)
            
            if results is not None:

                primary_leak = None
                for leak in results:
                    if leak.get('primary', False):
                        primary_leak = leak
                        break
                        
                all_results.append({
                    'file': os.path.basename(scenario_path),
                    'leaks': results,
                    'primary_leak': primary_leak
                })
        

        summary_path = os.path.join(output_dir, "leak_detection_summary.txt")
        with open(summary_path, 'w', encoding="utf-8") as f:
            f.write("ENHANCED LEAK DETECTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Sampling rate: {detector.sampling_rate/1e6:.1f} MS/s\n")
            f.write(f"Effective sampling rate: {detector.effective_sampling_rate/1e3:.1f} kS/s\n")
            f.write(f"Pipe length: {detector.pipe_length} m\n")
            f.write(f"Speed of sound: {detector.speed_of_sound:.1f} m/s\n\n")
            
            f.write("DETECTION RESULTS:\n")
            f.write("=" * 50 + "\n")
            
            for result in all_results:
                f.write(f"\n{result['file']}:\n")
                
                if result['leaks']:
                    for i, leak in enumerate(result['leaks']):
                        primary_marker = "★ " if leak.get('primary', False) else "  "
                        f.write(f"{primary_marker}Leak {i+1}: {leak['distance']:.3f}m "
                            f"(Δt = {leak['time_diff']*1000:.2f}ms, evidence = {leak['evidence']:.3f})\n")
                else:
                    f.write("  No leaks detected\n")
                    
                if result['primary_leak']:
                    f.write(f"  → PRIMARY LEAK: {result['primary_leak']['distance']:.3f}m\n")
        
        print(f"\nAnalysis complete! Results saved to '{output_dir}'")
        print(f"Summary: {summary_path}")




if __name__ == "__main__":
    x = EnhancedLeakDetector()
    x.starter()