# HydroLeakDetector

An advanced hydrophone-based leak detection system for pipelines.
Uses signal processing, adaptive filtering, and echo analysis to detect and localize leaks with high accuracy.

- Features

- Load hydrophone recordings (baseline & test scenario)

- Adaptive bandpass filtering around pinger frequency

- Automatic calibration using end-cap reflection

- Leak detection via echo difference analysis

- Confidence score for each detection

- Detailed plots (time, frequency, spectrograms, histograms)

- Text summary of detected leaks

## Installation
```bash
git clone https://github.com/your-username/HydroLeakDetector.git
cd HydroLeakDetector
pip install -r requirements.txt

```

## Usage  

```python
from EnhancedLeakDetector import EnhancedLeakDetector

detector = EnhancedLeakDetector(
    sampling_rate=12_500_000,   # ADC sample rate in Hz
    pipe_length=1.0             # Pipe length in meters
)
```
## Configuration
```python
detector = EnhancedLeakDetector(
    sampling_rate=12_500_000,  # ADC sample rate (Hz)
    pipe_length=1.0,           # Pipe length (m)
    downsample_factor=100,     # Data reduction factor
    speed_of_sound=1480,       # Speed of sound in water (m/s)
    pinger_freq=10000          # Pinger frequency (Hz)
)
```






