import os
import zipfile
import shutil
import subprocess
import sys

def prepare_humanml3d_data(output_dir="./data"):
    """
    Downloads the pre-processed HumanML3D dataset (272-dim modern variation)
    from HuggingFace and formats it perfectly for your PyTorch DiT DataLoader.
    
    Directory structure created:
    output_dir/
    ├── poses/ (contains .npy files)
    └── texts/ (contains .txt files)
    
    Usage in Colab:
        import setup_humanml3d
        setup_humanml3d.prepare_humanml3d_data()
    """
    # 1. Ensure huggingface_hub is installed
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    repo_id = "lxxiao/272-dim-HumanML3D"
    
    poses_dir = os.path.join(output_dir, "poses")
    texts_dir = os.path.join(output_dir, "texts")
    
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    # 2. Download ZIP files from HuggingFace Dataset hub
    print(f"Downloading texts.zip from {repo_id}...")
    texts_zip_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="texts.zip")
    
    print(f"Downloading motion_data.zip from {repo_id}...")
    motion_zip_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="motion_data.zip")
    
    # 3. Extract Texts directly to texts/ directory (flattening folders)
    print(f"Extracting texts to {texts_dir}...")
    with zipfile.ZipFile(texts_zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith('.txt'):
                filename = os.path.basename(member)
                if not filename: 
                    continue
                source = zip_ref.open(member)
                target_path = os.path.join(texts_dir, filename)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
    # 4. Extract Poses directly to poses/ directory (flattening folders)
    print(f"Extracting poses to {poses_dir}...")
    with zipfile.ZipFile(motion_zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith('.npy'):
                filename = os.path.basename(member)
                if not filename: 
                    continue
                # Exclude Mean.npy and Std.npy if they are in the zip, unless you want them!
                if filename in ["Mean.npy", "Std.npy"]:
                    # We can save these in the root data folder instead of poses/
                    source = zip_ref.open(member)
                    target_path = os.path.join(output_dir, filename)
                    with open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                    continue
                
                source = zip_ref.open(member)
                target_path = os.path.join(poses_dir, filename)
                with open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
    print(f"\n✅ Done! Dataset is ready at '{output_dir}'.")
    print(f"Found {len(os.listdir(poses_dir))} pose arrays and {len(os.listdir(texts_dir))} text annotations.")
    
if __name__ == "__main__":
    prepare_humanml3d_data()
