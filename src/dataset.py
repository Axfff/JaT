import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# Set backend to soundfile to avoid torchcodec issues
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

# Normalization Constants (Fixed Global Statistics)
# Log-Mel Spectrogram values typically range from -11.0 to 0.0 (after log(spec + 1e-9))
# We center around -5.0 and scale by 3.0 to get approx [-2, 2] range, then clip and scale to [-1, 1]
NORM_MEAN = -5.0
NORM_STD = 3.0

# SpeechCommands 12 Core Classes (10 command words + unknown + silence)
# These are the standard evaluation classes for keyword spotting benchmarks
CORE_CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",  # 10 core commands
    "unknown",  # All other words (bed, bird, cat, dog, etc.)
    "silence"   # Background noise / silence
]

def normalize_spectrogram(spec):
    """
    Normalize spectrogram using fixed global statistics.
    Args:
        spec: (..., F, T) Mel spectrogram (log-scaled)
    Returns:
        norm_spec: Normalized spectrogram in range approx [-1, 1]
    """
    # Standardize
    norm_spec = (spec - NORM_MEAN) / NORM_STD
    
    # Clip to [-3, 3] (approx 99.7% of data if Gaussian, but here just safety)
    norm_spec = torch.clamp(norm_spec, min=-3.0, max=3.0)
    
    # Scale to [-1, 1]
    norm_spec = norm_spec / 3.0
    
    return norm_spec

def denormalize_spectrogram(norm_spec):
    """
    Denormalize spectrogram to original log-scale range.
    Args:
        norm_spec: (..., F, T) Normalized spectrogram in range [-1, 1]
    Returns:
        spec: Log-scaled Mel spectrogram
    """
    # Rescale from [-1, 1] to [-3, 3]
    spec = norm_spec * 3.0
    
    # Destandardize
    spec = spec * NORM_STD + NORM_MEAN
    
    return spec

class SpeechCommandsDataset(Dataset):
    def __init__(self, root, mode='raw', subset='training', download=True, mock=False, freq_bins=64, time_frames=64, core_classes_only=False):
        """
        Args:
            root (str): Root directory for the dataset.
            mode (str): 'raw' or 'spectrogram'.
            subset (str): 'training', 'validation', or 'testing'.
            download (bool): Whether to download the dataset.
            mock (bool): Whether to use mock data.
            freq_bins (int): Number of mel frequency bins.
            time_frames (int): Number of time frames.
            core_classes_only (bool): If True, use only 12 core classes 
                (yes, no, up, down, left, right, on, off, stop, go, unknown, silence).
                Non-core words are mapped to 'unknown', silence samples from 
                _background_noise_ are mapped to 'silence'.
        """
        self.mode = mode
        self.subset = subset
        self.sample_rate = 16000
        self.duration = 1.0 # seconds
        self.duration = 1.0 # seconds
        self.num_samples = 16384 # Pad to multiple of 512 (32 * 512)
        self.mock = mock
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.core_classes_only = core_classes_only
        
        # Define the 10 core command words (used for mapping)
        self.core_commands = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}
        
        if self.mock:
            self.dataset = list(range(100)) # 100 mock samples
            self._indices = list(range(100))
        else:
            # Create the dataset
            # Note: 'speech_commands_v0.02' is the v2 dataset
            self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
                root=root,
                url='speech_commands_v0.02',
                download=download,
                subset=None
            )
            
            # Load list of files for the requested subset
            self._indices = self._get_subset_indices()
        
        # Build label map
        if self.mock:
            if core_classes_only:
                self.all_labels = CORE_CLASSES
            else:
                self.all_labels = ["bed", "cat", "down"]
        else:
            if core_classes_only:
                # Use only 12 core classes
                self.all_labels = CORE_CLASSES
            else:
                # Use all 35 labels
                # We need to scan the dataset to find all labels. 
                # This might be slow if we iterate all.
                # But SpeechCommands structure is root/label/file.
                # We can list directories in root/SpeechCommands/speech_commands_v0.02
                # self.dataset._path is the path.
                dataset_path = self.dataset._path
                self.all_labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and d != '_background_noise_'])
            
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}

    def _get_subset_indices(self):
        # This is a simplified split logic. 
        # For a rigorous research, we should parse 'validation_list.txt' and 'testing_list.txt'.
        # However, torchaudio.datasets.SPEECHCOMMANDS abstracts the file paths.
        # Let's see if we can access the file list.
        
        # Actually, let's implement a standard split:
        # The dataset object behaves like a list.
        # We can iterate over it to get metadata, but that's slow.
        
        # Alternative: Use the standard validation/testing list provided by the dataset.
        # The dataset is downloaded to root/SpeechCommands/speech_commands_v0.02/
        
        dataset_path = os.path.join(self.dataset._path)
        val_list_path = os.path.join(dataset_path, "validation_list.txt")
        test_list_path = os.path.join(dataset_path, "testing_list.txt")
        
        # If files don't exist yet (before download), we can't split. 
        # But download happens in __init__.
        
        if not os.path.exists(val_list_path):
            # If dataset was just downloaded, these should exist.
            # If not, maybe we are in a weird state. 
            # For now, let's assume we use all data if split files are missing (or handle gracefully).
            return range(len(self.dataset))

        with open(val_list_path, 'r') as f:
            val_files = set(x.strip() for x in f.readlines())
            
        with open(test_list_path, 'r') as f:
            test_files = set(x.strip() for x in f.readlines())
            
        indices = []
        for i in range(len(self.dataset)):
            # The dataset item is (waveform, sample_rate, label, speaker_id, utterance_number)
            # But getting the item loads the audio, which is slow for just splitting.
            # We need the relative path.
            # torchaudio 0.12+ structure:
            # self.dataset._walker is a list of file paths (relative or absolute depending on version)
            
            file_path = self.dataset._walker[i]
            # file_path is usually something like "bed/00176480_nohash_0.wav" or absolute path.
            # We need to normalize it to match the list.
            
            rel_path = os.path.relpath(file_path, dataset_path)
            
            if self.subset == 'validation':
                if rel_path in val_files:
                    indices.append(i)
            elif self.subset == 'testing':
                if rel_path in test_files:
                    indices.append(i)
            elif self.subset == 'training':
                if rel_path not in val_files and rel_path not in test_files:
                    indices.append(i)
        
        return indices
    
    def _map_to_core_class(self, label):
        """
        Map a label to one of the 12 core classes.
        
        Args:
            label: Original label string (e.g., "yes", "bed", "_background_noise_")
            
        Returns:
            Mapped label string (one of CORE_CLASSES)
        """
        if label in self.core_commands:
            # Core command word - keep as-is
            return label
        elif label == '_background_noise_' or label == 'silence':
            # Background noise / silence
            return 'silence'
        else:
            # All other words map to 'unknown'
            return 'unknown'

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        if self.mock:
            # Generate random waveform
            waveform = torch.randn(1, self.num_samples)
            label = "bed" # Mock label
            # Map to core class if needed
            if self.core_classes_only:
                label = self._map_to_core_class(label)
            label_idx = self.label_to_idx.get(label, 0)
            
            if self.mode in ['spectrogram', 'spectrum_1d']:
                 # Mock spectrogram
                 spec = torch.randn(self.freq_bins, self.time_frames)
                 return spec, label_idx
            return waveform, label_idx

        original_idx = self._indices[idx]
        # waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[original_idx]
        # Manual load to bypass torchcodec
        file_path = self.dataset._walker[original_idx]
        # file_path is absolute or relative?
        # In torchaudio 0.12+, _walker contains full paths usually.
        # Let's verify if it's absolute.
        # If not, join with root.
        # But wait, self.dataset is SPEECHCOMMANDS object.
        # Let's assume it works or try to load.
        
        # We need label.
        # Label is directory name.
        rel_path = os.path.relpath(file_path, self.dataset._path)
        label = os.path.split(rel_path)[0]
        
        # Map to core class if needed
        if self.core_classes_only:
            label = self._map_to_core_class(label)
        
        # Load audio
        # waveform, sample_rate = torchaudio.load(file_path, backend="soundfile")
        # Or use soundfile directly
        import soundfile as sf
        wav_np, sample_rate = sf.read(file_path)
        waveform = torch.from_numpy(wav_np).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # [1, T]
        
        # Resample if necessary (though SpeechCommands is usually 16k)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        # Pad or Crop to 1 second
        if waveform.shape[1] < self.num_samples:
            pad_amount = self.num_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount))
        elif waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
            
        # Normalize to [-1, 1] (Usually it is already float32 in [-1, 1] when loaded by torchaudio)
        # But let's ensure it.
        
        if self.mode in ['spectrogram', 'spectrum_1d']:
            # Convert to Mel Spectrogram (both modes use 2D spectrograms, differ in model patchification)
            # Target: freq_bins x time_frames
            
            # Dynamically calculate settings
            # n_fft depends on freq_bins (n_mels). 
            # If freq_bins is large (e.g. 128), n_fft=1024 is fine (513 bins). 
            # If freq_bins > 512, we might need larger n_fft.
            n_fft = 2048 if self.freq_bins > 512 else 1024
            
            # Hop length to approximate time_frames
            # We want roughly 'time_frames' columns.
            hop_length = self.num_samples // self.time_frames
            
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length, 
                n_mels=self.freq_bins
            )
            spec = mel_transform(waveform)
            # spec shape: [1, freq_bins, T_actual]
            
            # Resize to exactly freq_bins x time_frames
            # We use interpolate to ensure exact dimensions regardless of rounding in STFT
            spec = F.interpolate(spec.unsqueeze(0), size=(self.freq_bins, self.time_frames), mode='bilinear', align_corners=False).squeeze(0)
            
            # Log scaling usually helps
            spec = torch.log(spec + 1e-9)
            
            # Normalize using fixed global statistics
            spec = normalize_spectrogram(spec)
            
            return spec, self.label_to_idx.get(label, 0)
            
        return waveform, self.label_to_idx.get(label, 0)

def get_dataloader(root, mode='raw', subset='training', batch_size=32, num_workers=4, mock=False, freq_bins=64, time_frames=64, core_classes_only=False):
    dataset = SpeechCommandsDataset(root, mode=mode, subset=subset, mock=mock, freq_bins=freq_bins, time_frames=time_frames, core_classes_only=core_classes_only)
    
    # We need to collate labels properly because they are strings.
    # Or we can convert them to indices.
    # For now, let's return strings or handle it.
    # Standard practice: create a label-to-index mapping.
    
    # Let's add a simple collate_fn or wrapper if we want indices.
    # But the dataset returns (data, label_string).
    # Let's assume the training loop handles string labels or we map them here.
    # For simplicity, let's map them to indices in the dataset.
    
    # Actually, let's just return the dataset and let the user wrap it in DataLoader
    # or return the DataLoader directly.
    
    # To make it robust, let's build a label map.
    # The dataset now returns integer labels directly, so no custom collate_fn is needed.
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(subset == 'training'),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

if __name__ == "__main__":
    # Test the dataset
    root_dir = "./data"
    os.makedirs(root_dir, exist_ok=True)
    
    print("Creating dataset...")
    ds = SpeechCommandsDataset(root_dir, mode='raw', subset='training', download=True)
    print(f"Dataset size: {len(ds)}")
    
    x, y = ds[0]
    print(f"Sample 0: Shape={x.shape}, Label={y}")
    
    dl = get_dataloader(root_dir, mode='spectrogram', subset='validation', batch_size=4)
    for x, y in dl:
        print(f"Batch shape: {x.shape}, Labels: {y}")
        break
