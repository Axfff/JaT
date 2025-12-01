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

class SpeechCommandsDataset(Dataset):
    def __init__(self, root, mode='raw', subset='training', download=True, mock=False):
        """
        Args:
            root (str): Root directory for the dataset.
            mode (str): 'raw' or 'spectrogram'.
            subset (str): 'training', 'validation', or 'testing'.
            download (bool): Whether to download the dataset.
            mock (bool): Whether to use mock data.
        """
        self.mode = mode
        self.subset = subset
        self.sample_rate = 16000
        self.duration = 1.0 # seconds
        self.num_samples = 16384 # Pad to multiple of 512 (32 * 512)
        self.mock = mock
        
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
            self.all_labels = ["bed", "cat", "down"]
        else:
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

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        if self.mock:
            # Generate random waveform
            waveform = torch.randn(1, self.num_samples)
            label = "bed" # Mock label
            label_idx = self.label_to_idx.get(label, 0)
            
            if self.mode == 'spectrogram':
                 # Mock spectrogram
                 spec = torch.randn(64, 64)
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
        
        if self.mode == 'spectrogram':
            # Convert to Mel Spectrogram
            # Target: 64x64
            # 16000 samples. 
            # To get 64 time steps: hop_length = 16000 / 64 = 250.
            # n_fft needs to be appropriate.
            
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=256, # 16000/256 = 62.5 -> approx 63 frames. Close to 64.
                n_mels=64
            )
            spec = mel_transform(waveform)
            # spec shape: [1, 64, T]
            
            # Resize to exactly 64x64 if needed
            spec = F.interpolate(spec.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
            
            # Log scaling usually helps
            spec = torch.log(spec + 1e-9)
            
            # Normalize to zero mean and unit variance (approximate based on batch statistics or fixed)
            # For simplicity and stability, let's just normalize per sample for now, 
            # or use a fixed global mean/std if known. 
            # Let's use per-sample standardization.
            spec = (spec - spec.mean()) / (spec.std() + 1e-5)
            
            return spec, self.label_to_idx.get(label, 0)
            
        return waveform, self.label_to_idx.get(label, 0)

def get_dataloader(root, mode='raw', subset='training', batch_size=32, num_workers=4, mock=False):
    dataset = SpeechCommandsDataset(root, mode=mode, subset=subset, mock=mock)
    
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
