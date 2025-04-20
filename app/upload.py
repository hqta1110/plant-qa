from huggingface_hub import HfApi, upload_file, login
import os

# Login to Hugging Face (will prompt for token)
# You can get your token from https://huggingface.co/settings/tokens
login()

# Initialize the Hugging Face API
api = HfApi()

# Define your repository name (create this repo on Hugging Face first)
repo_id = "hqta1110/plant-classification-models"

# Create the repository if it doesn't exist 
# (you can also create it manually on the website)
try:
    api.create_repo(repo_id=repo_id, private=False)
    print(f"Created repository: {repo_id}")
except Exception as e:
    print(f"Repository might already exist or there was an error: {e}")

# Path to your model files
model_files = [
    "/home/sora/pretrain-llm/infer/models/dino_best.pth",
    "/home/sora/pretrain-llm/infer/models/embedding_best.pth"
]

# Upload each model file
for model_path in model_files:
    print(f"Uploading {model_path}...")
    
    # Get just the filename without the path
    filename = os.path.basename(model_path)
    
    # Upload the file
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {filename} to {repo_id}")

# Path to your index file and metadata
additional_files = [
    "/home/sora/pretrain-llm/infer/data/plant_index_2.pkl",
    "/home/sora/pretrain-llm/infer/data/merge_metadata.json"
]

# Upload additional files
for file_path in additional_files:
    print(f"Uploading {file_path}...")
    
    # Get just the filename without the path
    filename = os.path.basename(file_path)
    
    # Upload the file
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {filename} to {repo_id}")

print("All files uploaded successfully!")