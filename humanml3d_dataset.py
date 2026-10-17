import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HumanML3DDataset(Dataset):
    def __init__(self, data_dir, max_frames=200):
        """
        data_dir: Path to the directory containing 'poses' and 'texts' subfolders.
        max_frames: The fixed sequence length for the Transformer. 
                    Sequences longer than this are cropped. Sequences shorter are padded with zeros.
        """
        self.poses_dir = os.path.join(data_dir, "poses")
        self.texts_dir = os.path.join(data_dir, "texts")
        self.max_frames = max_frames
        
        # Find all valid paired files
        pose_files = [f for f in os.listdir(self.poses_dir) if f.endswith('.npy')]
        self.samples = []
        
        for p_file in pose_files:
            # Ensure we skip Mean.npy and Std.npy if they ended up here
            if p_file in ["Mean.npy", "Std.npy"]:
                continue
                
            file_id = os.path.splitext(p_file)[0]
            t_file = f"{file_id}.txt"
            
            # Check if matching text file exists
            if os.path.exists(os.path.join(self.texts_dir, t_file)):
                self.samples.append(file_id)
                
        # Sort for deterministic ordering
        self.samples.sort()
        print(f"Found {len(self.samples)} valid pose-text pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id = self.samples[idx]
        
        # ----------------------------------------------------
        # 1. LOAD THE SKELETON TENSOR
        # ----------------------------------------------------
        pose_path = os.path.join(self.poses_dir, f"{file_id}.npy")
        skeleton = np.load(pose_path).astype(np.float32) # Shape: [Num_Frames, 272]
        
        # Get actual number of frames before we start padding
        num_frames = skeleton.shape[0]
        
        # Transformers need a fixed size (e.g., 200). We must Pad or Crop.
        if num_frames > self.max_frames:
            # Crop to max_frames
            skeleton = skeleton[:self.max_frames, :]
            valid_len = self.max_frames
        else:
            # Zero-pad the remaining frames
            pad_len = self.max_frames - num_frames
            padding = np.zeros((pad_len, skeleton.shape[1]), dtype=np.float32)
            skeleton = np.concatenate([skeleton, padding], axis=0)
            valid_len = num_frames
            
        # Convert to PyTorch Tensor
        skeleton_tensor = torch.tensor(skeleton)
        
        # ----------------------------------------------------
        # 2. LOAD & AUGMENT THE TEXT
        # ----------------------------------------------------
        text_path = os.path.join(self.texts_dir, f"{file_id}.txt")
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Parse the custom HumanML3D format: original_desc#processed_desc#start#end
        valid_texts = []
        for line in lines:
            parts = line.split('#')
            if len(parts) > 0:
                valid_texts.append(parts[0].strip()) # Grab just the natural English sentence
                
        if not valid_texts:
            valid_texts = ["A person is moving."] # Safe fallback
            
        # TEXT AUGMENTATION: Randomly pick one description for this epoch!
        selected_text = random.choice(valid_texts)
        
        return {
            "skeleton": skeleton_tensor,
            "text": selected_text,
            "valid_len": torch.tensor(valid_len, dtype=torch.long)
        }

def get_dataloader(data_dir, batch_size=32, max_frames=200, shuffle=True, num_workers=2):
    """
    Helper function to initialize the dataset and wrap it in a DataLoader.
    """
    dataset = HumanML3DDataset(data_dir=data_dir, max_frames=max_frames)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=True, # Drops the last incomplete batch for stable training
        pin_memory=True,
        prefetch_factor=2
    )
    return dataloader

# ==========================================
# TEST THE DATALOADER
# ==========================================
if __name__ == "__main__":
    # Test path (Make sure you ran setup_humanml3d.py first!)
    TEST_DIR = "my_pose_dit_project/data"
    
    if os.path.exists(TEST_DIR):
        print(f"Testing DataLoader from {TEST_DIR}...\n")
        dl = get_dataloader(TEST_DIR, batch_size=4, max_frames=120)
        
        for batch in dl:
            print("--- BATCH SHAPES ---")
            print("Skeleton Tensor Shape:", batch["skeleton"].shape) # Expected: (4, 120, 272)
            print("Valid Lengths (before padding):", batch["valid_len"].tolist())
            
            print("\n--- BATCH TEXTS ---")
            for i, text in enumerate(batch["text"]):
                print(f"Sample {i+1}: {text}")
            break
    else:
        print(f"Error: {TEST_DIR} does not exist. Please run setup_humanml3d.py first.")
