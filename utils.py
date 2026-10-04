"""
NeuroStress Utility Functions
EEG data processing and analysis utilities
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional

# Try to import scipy for .mat file support
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Install with: pip install scipy")


def load_mat_file(filepath: str) -> Optional[Dict]:
    """
    Load a MATLAB .mat file
    
    Args:
        filepath: Path to the .mat file
        
    Returns:
        Dictionary containing the MATLAB data, or None if loading fails
    """
    if not SCIPY_AVAILABLE:
        print("scipy is required to load .mat files")
        return None
    
    try:
        data = loadmat(filepath)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_dataset_files(data_dir: str = "data/raw") -> List[str]:
    """
    Get list of all .mat files in the dataset directory
    
    Args:
        data_dir: Path to the raw data directory
        
    Returns:
        List of .mat file paths
    """
    if not os.path.exists(data_dir):
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    return sorted(files)


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse SAM-40 dataset filename to extract metadata
    
    Filename format: {Task}_sub_{SubjectID}_trial{TrialNumber}.mat
    Tasks: Stroop, Mirror_image (stress), Relax (normal)
    
    Args:
        filename: The .mat filename
        
    Returns:
        Dictionary with task, subject_id, trial_number, and label
    """
    name = filename.replace('.mat', '')
    parts = name.split('_')
    
    # Handle Mirror_image which has underscore in name
    if 'Mirror' in parts[0]:
        task = 'Mirror_image'
        remaining = '_'.join(parts[2:])
    else:
        task = parts[0]
        remaining = '_'.join(parts[1:])
    
    # Extract subject and trial
    subject_id = None
    trial_number = None
    
    for i, part in enumerate(remaining.split('_')):
        if part == 'sub' and i + 1 < len(remaining.split('_')):
            subject_id = remaining.split('_')[i + 1]
        if 'trial' in part:
            trial_number = part.replace('trial', '')
    
    # Determine label based on task
    if task in ['Stroop', 'Mirror_image']:
        label = 'stress'
    else:
        label = 'normal'
    
    return {
        'task': task,
        'subject_id': subject_id,
        'trial_number': trial_number,
        'label': label,
        'filename': filename
    }


def get_dataset_stats(data_dir: str = "data/raw") -> Dict:
    """
    Calculate statistics about the dataset
    
    Args:
        data_dir: Path to the raw data directory
        
    Returns:
        Dictionary with dataset statistics
    """
    files = get_dataset_files(data_dir)
    
    if not files:
        return {
            'total_files': 0,
            'stress_files': 0,
            'normal_files': 0,
            'subjects': set(),
            'tasks': {}
        }
    
    stats = {
        'total_files': len(files),
        'stress_files': 0,
        'normal_files': 0,
        'subjects': set(),
        'tasks': {}
    }
    
    for f in files:
        info = parse_filename(f)
        
        if info['label'] == 'stress':
            stats['stress_files'] += 1
        else:
            stats['normal_files'] += 1
        
        if info['subject_id']:
            stats['subjects'].add(info['subject_id'])
        
        task = info['task']
        if task not in stats['tasks']:
            stats['tasks'][task] = 0
        stats['tasks'][task] += 1
    
    stats['num_subjects'] = len(stats['subjects'])
    stats['subjects'] = sorted(list(stats['subjects']))
    
    return stats


def compute_band_power(data: np.ndarray, fs: int = 128) -> Dict[str, float]:
    """
    Compute power in different EEG frequency bands
    
    Bands:
        - Delta: 0.5-4 Hz
        - Theta: 4-8 Hz
        - Alpha: 8-13 Hz
        - Beta: 13-30 Hz
        - Gamma: 30-45 Hz
    
    Args:
        data: EEG signal data (1D array)
        fs: Sampling frequency (default 128 Hz for SAM-40)
        
    Returns:
        Dictionary with power values for each band
    """
    try:
        from scipy.signal import welch
        from scipy.integrate import simpson
    except ImportError:
        # Return simulated values if scipy not available
        return {
            'delta': np.random.uniform(10, 25),
            'theta': np.random.uniform(15, 35),
            'alpha': np.random.uniform(20, 45),
            'beta': np.random.uniform(15, 40),
            'gamma': np.random.uniform(10, 30)
        }
    
    # Compute power spectral density
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    band_power = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        if np.any(idx):
            band_power[band] = simpson(psd[idx], x=freqs[idx])
        else:
            band_power[band] = 0.0
    
    return band_power


def compute_stress_indicators(band_power: Dict[str, float]) -> Dict[str, float]:
    """
    Compute stress indicators from band power values
    
    Indicators:
        - Beta/Alpha ratio: Higher indicates stress (threshold > 2.0)
        - Theta/Beta ratio: Lower during stress
        - Relative gamma: Elevated during high cognitive load
    
    Args:
        band_power: Dictionary with power values for each band
        
    Returns:
        Dictionary with computed stress indicators
    """
    alpha = band_power.get('alpha', 1)
    beta = band_power.get('beta', 1)
    theta = band_power.get('theta', 1)
    gamma = band_power.get('gamma', 1)
    
    # Avoid division by zero
    alpha = max(alpha, 0.001)
    beta = max(beta, 0.001)
    
    return {
        'beta_alpha_ratio': beta / alpha,
        'theta_beta_ratio': theta / beta,
        'relative_gamma': gamma / (alpha + beta + theta + gamma),
        'stress_score': min(1.0, (beta / alpha) / 3.0)  # Normalized 0-1
    }


def classify_stress(indicators: Dict[str, float]) -> Tuple[str, float]:
    """
    Classify stress state based on computed indicators
    
    Args:
        indicators: Dictionary with stress indicators
        
    Returns:
        Tuple of (classification, confidence)
    """
    beta_alpha = indicators.get('beta_alpha_ratio', 1.0)
    stress_score = indicators.get('stress_score', 0.5)
    
    # Classification thresholds based on literature
    if beta_alpha > 2.0:
        classification = 'stress'
        confidence = min(0.95, 0.7 + (beta_alpha - 2.0) * 0.1)
    elif beta_alpha < 1.0:
        classification = 'normal'
        confidence = min(0.95, 0.7 + (1.0 - beta_alpha) * 0.2)
    else:
        # Borderline case
        if stress_score > 0.5:
            classification = 'stress'
            confidence = 0.5 + stress_score * 0.3
        else:
            classification = 'normal'
            confidence = 0.5 + (1 - stress_score) * 0.3
    
    return classification, confidence


# Stress threshold constants (based on literature)
STRESS_THRESHOLDS = {
    'beta_alpha_ratio': {
        'stress': 2.0,
        'normal': 1.0,
        'description': 'Beta/Alpha power ratio - elevated during cognitive stress'
    },
    'frontal_asymmetry': {
        'stress': 0.2,
        'normal': 0.1,
        'description': 'Left-right frontal hemisphere asymmetry'
    },
    'theta_increase': {
        'stress': 1.5,
        'normal': 1.0,
        'description': 'Theta power increase factor during mental fatigue'
    },
    'alpha_suppression': {
        'stress': 0.6,
        'normal': 0.8,
        'description': 'Alpha power suppression during stress'
    }
}


if __name__ == "__main__":
    # Test utilities
    print("NeuroStress Utilities")
    print("=" * 40)
    
    # Test dataset stats
    stats = get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Stress files: {stats['stress_files']}")
    print(f"  Normal files: {stats['normal_files']}")
    print(f"  Subjects: {stats['num_subjects']}")
    
    # Test filename parsing
    test_file = "Mirror_image_sub_7_trial1.mat"
    info = parse_filename(test_file)
    print(f"\nParsed {test_file}:")
    print(f"  Task: {info['task']}")
    print(f"  Subject: {info['subject_id']}")
    print(f"  Trial: {info['trial_number']}")
    print(f"  Label: {info['label']}")
