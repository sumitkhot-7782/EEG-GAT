"""
NeuroStress Data Loader
Utilities for loading and preprocessing SAM-40 EEG dataset
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from config import RAW_DATA_DIR, SAMPLING_RATE, FREQUENCY_BANDS, TASK_LABELS

# Try to import scipy for .mat file support
try:
    from scipy.io import loadmat
    from scipy.signal import welch, butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SAM40DataLoader:
    """
    Data loader for SAM-40 EEG dataset
    
    The SAM-40 dataset contains EEG recordings from 40 subjects
    performing cognitive stress tasks and relaxation.
    
    Tasks:
        - Stroop: Color-word interference (stress)
        - Mirror_image: Mirror image recognition (stress)
        - Arithmetic: Mental arithmetic (stress)
        - Relax: Eyes closed relaxation (normal)
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to the raw data directory
        """
        self.data_dir = data_dir or RAW_DATA_DIR
        self.files = []
        self.metadata = []
        
    def scan_dataset(self) -> List[Dict]:
        """
        Scan the dataset directory and extract file metadata
        
        Returns:
            List of file metadata dictionaries
        """
        self.files = []
        self.metadata = []
        
        if not os.path.exists(self.data_dir):
            print(f"Data directory not found: {self.data_dir}")
            return []
        
        for filename in sorted(os.listdir(self.data_dir)):
            if filename.endswith('.mat'):
                self.files.append(filename)
                self.metadata.append(self._parse_filename(filename))
        
        return self.metadata
    
    def _parse_filename(self, filename: str) -> Dict:
        """
        Parse SAM-40 filename to extract metadata
        
        Args:
            filename: The .mat filename
            
        Returns:
            Metadata dictionary
        """
        name = filename.replace('.mat', '')
        parts = name.split('_')
        
        # Handle Mirror_image which has underscore
        if 'Mirror' in parts[0]:
            task = 'Mirror_image'
            remaining = parts[2:]
        else:
            task = parts[0]
            remaining = parts[1:]
        
        subject_id = None
        trial = None
        
        for i, part in enumerate(remaining):
            if part == 'sub' and i + 1 < len(remaining):
                subject_id = remaining[i + 1]
            if 'trial' in part:
                trial = part.replace('trial', '')
        
        label = TASK_LABELS.get(task, 'unknown')
        
        return {
            'filename': filename,
            'filepath': os.path.join(self.data_dir, filename),
            'task': task,
            'subject_id': subject_id,
            'trial': trial,
            'label': label,
            'label_int': 1 if label == 'stress' else 0
        }
    
    def load_file(self, filename: str) -> Optional[Dict]:
        """
        Load a single .mat file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary containing the EEG data
        """
        if not SCIPY_AVAILABLE:
            print("scipy is required to load .mat files")
            return None
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data = loadmat(filepath)
            return data
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Statistics dictionary
        """
        if not self.metadata:
            self.scan_dataset()
        
        stats = {
            'total_files': len(self.files),
            'stress_files': sum(1 for m in self.metadata if m['label'] == 'stress'),
            'normal_files': sum(1 for m in self.metadata if m['label'] == 'normal'),
            'subjects': len(set(m['subject_id'] for m in self.metadata if m['subject_id'])),
            'tasks': {}
        }
        
        for m in self.metadata:
            task = m['task']
            if task not in stats['tasks']:
                stats['tasks'][task] = 0
            stats['tasks'][task] += 1
        
        return stats
    
    def create_train_test_split(self, train_ratio: float = 0.8,
                                val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """
        Create train/validation/test split
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        if not self.metadata:
            self.scan_dataset()
        
        # Shuffle files
        indices = np.random.permutation(len(self.files))
        
        n_train = int(len(self.files) * train_ratio)
        n_val = int(len(self.files) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_files = [self.metadata[i] for i in train_idx]
        val_files = [self.metadata[i] for i in val_idx]
        test_files = [self.metadata[i] for i in test_idx]
        
        return train_files, val_files, test_files


def preprocess_eeg(data: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Preprocess EEG signal
    
    Steps:
    1. Bandpass filter (0.5-45 Hz)
    2. Normalize to zero mean, unit variance
    
    Args:
        data: Raw EEG data (channels x samples)
        fs: Sampling frequency
        
    Returns:
        Preprocessed EEG data
    """
    if not SCIPY_AVAILABLE:
        # Return normalized data without filtering
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    # Bandpass filter 0.5-45 Hz
    nyq = fs / 2
    low = 0.5 / nyq
    high = 45 / nyq
    
    b, a = butter(4, [low, high], btype='band')
    
    # Apply filter to each channel
    if len(data.shape) == 1:
        filtered = filtfilt(b, a, data)
    else:
        filtered = np.apply_along_axis(lambda x: filtfilt(b, a, x), 1, data)
    
    # Normalize
    normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    return normalized


def compute_spectrogram(data: np.ndarray, fs: int = SAMPLING_RATE,
                        window_size: int = 256) -> np.ndarray:
    """
    Compute spectrogram from EEG data
    
    Args:
        data: EEG signal (1D array)
        fs: Sampling frequency
        window_size: FFT window size
        
    Returns:
        Spectrogram array (frequency x time)
    """
    if not SCIPY_AVAILABLE:
        # Return simulated spectrogram
        return np.random.rand(64, 128)
    
    from scipy.signal import spectrogram as scipy_spectrogram
    
    f, t, Sxx = scipy_spectrogram(data, fs=fs, nperseg=window_size,
                                   noverlap=window_size//2)
    
    # Keep frequencies up to 45 Hz
    freq_mask = f <= 45
    
    return Sxx[freq_mask, :]


if __name__ == "__main__":
    # Test data loader
    print("Testing SAM-40 Data Loader")
    print("=" * 50)
    
    loader = SAM40DataLoader()
    metadata = loader.scan_dataset()
    
    print(f"\nFound {len(metadata)} files")
    
    stats = loader.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Stress files: {stats['stress_files']}")
    print(f"  Normal files: {stats['normal_files']}")
    print(f"  Subjects: {stats['subjects']}")
    print(f"  Tasks: {stats['tasks']}")
    
    if metadata:
        print(f"\nSample file metadata:")
        print(f"  {metadata[0]}")
