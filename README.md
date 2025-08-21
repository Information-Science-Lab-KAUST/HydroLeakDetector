# HydroLeakDetector

**HydroLeakDetector** is an advanced signal processing tool for locating leaks in water pipelines using hydrophone sensor data. It applies enhanced acoustic analysis to differentiate between baseline (no-leak) and scenario (potential leak) measurements, estimates leak positions, and generates detailed diagnostics and visualizations.

## Features

- **Automated Data Loading & Preprocessing:** Handles large CSV files, downsampling for faster computation.
- **Adaptive Filtering:** Dynamic bandpass filter tuning based on signal characteristics.
- **Pulse Detection & Calibration:** Identifies direct acoustic pulses and calibrates the speed of sound using end-cap reflections.
- **Leak Echo Detection:** Detects and ranks leak echoes based on multiple evidence metrics.
- **Visualization:** Generates comprehensive plots for signal comparison, differences, spectrograms, and leak distance estimation.
- **Batch Analysis Support:** Processes multiple scenario files against a baseline.
- **Summary Reports:** Outputs a human-readable summary of detected leaks and analysis results.

## Usage

### 1. Prepare Your Data

- **Baseline file:** CSV file with columns for time and signal (no leak present).
- **Scenario files:** CSV files (same format) representing measurements with potential leaks.
- Place these files in your designated folder.

### 2. Configure Paths

Edit the `main` function in `Main.py` to specify:
- `baseline_path`: Path to your baseline CSV file.
- `output_dir`: Directory where results and plots will be saved.
- `scenario_files`: Set the glob pattern for your scenario data files.

Example:
```python
baseline_path = "C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\Response Folder\\analogbaseline.csv"
output_dir = "C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\enhanced_analysis"
scenario_files = glob.glob("C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\Response Folder\\analog.csv")
```

### 3. Run the Program

Ensure your environment has the required dependencies (see below), then run:

```bash
python Main.py
```

### 4. View Results

- **Plots:** Saved in the `output_dir`, showing in-depth analysis for each scenario.
- **Summary:** A `leak_detection_summary.txt` file summarizing leak detections.

## Dependencies

- Python 3.6+
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scipy](https://scipy.org/)

Install requirements with:

```bash
pip install numpy pandas matplotlib scipy
```

## Example Output

```
=== RESULTS FOR analog.csv ===
Detected 1 potential echoes:
★ Echo 1:
   Distance: 0.420 m from sensor
   Time delay: 0.57 ms
   Evidence score: 1.354
   STATUS: PRIMARY LEAK CANDIDATE

=== VALIDATION ===
Primary leak detected at: 0.420 m
Time delay: 0.57 ms
Evidence score: 1.354
✓ Distance is within pipe bounds (0 - 1.0 m)
✓ Strong evidence for leak detection
```

## Customization

- **Sampling Rate & Pipe Length:** Change in `EnhancedLeakDetector` constructor.
- **Signal Processing Parameters:** Adjust bandpass filter, window sizes, and detection thresholds in class methods for different setups or pipe materials.

## Notes

- This tool is designed for research and development. Real-world performance may require parameter tuning and validation with your specific hardware and environment.
- For best results, ensure your baseline and scenario measurements are properly aligned and acquired under similar conditions.

## License

[MIT License](LICENSE)

## Contact

For questions or collaboration, please contact the [Information Science Lab, KAUST](https://github.com/Information-Science-Lab-KAUST).
