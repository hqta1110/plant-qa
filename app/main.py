from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
import os
import tempfile
from PIL import Image
import uvicorn
import shutil
import json

# Import your model components
from classifier import HybridClassifier
from simImage import SimpleClassificationPipeline
from llm_huggingface import PlantQA
from embedder import ImageEmbeddingModel
from retrieval_system import ImageRetrievalSystem, FeatureExtractor
import os
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  
app = FastAPI(title="Plant Classification and Q&A API", 
              description="API for classifying plants and answering questions about them")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths - update these to your actual paths
MODEL_CONFIG = {
    "classifier_path": os.environ.get("CLASSIFIER_PATH", "/home/sora/pretrain-llm/infer/models/dino_best.pth"),
    "embed_path": os.environ.get("EMBED_PATH", "/home/sora/pretrain-llm/infer/models/embedding_best.pth"), 
    "labels_path": os.environ.get("LABELS_PATH", "/home/sora/pretrain-llm/data/data"),
    "metadata_path": os.environ.get("METADATA_PATH", "/home/sora/pretrain-llm/infer/data/merge_metadata.json"),
    "retrieval_index_file": os.environ.get("RETRIEVAL_INDEX_PATH", "/home/sora/pretrain-llm/infer/data/plant_index_2.pkl")
}

# Global variables for models
classification_pipeline = None
llm_qa = None
retrieval_system = None
plant_metadata = None

# Temporary directory for uploaded images
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def download_model_files():
    """
    Downloads all required model files from Hugging Face Hub.
    Creates the necessary directories and returns the paths to the files.
    """
    print("Checking and downloading model files...")
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Your Hugging Face repository ID
    repo_id = "hqta1110/plant-classification-models"
    
    # Define the files to download with their destinations
    files_to_download = {
        "dino_best.pth": "../models/dino_best.pth",
        "embedding_best.pth": "../models/embedding_best.pth",
        "plant_index_2.pkl": "../data/plant_index_2.pkl",
        "merge_metadata.json": "../data/merge_metadata.json"
    }
    
    downloaded_paths = {}
    
    # Download each file individually
    for filename, destination in files_to_download.items():
        if not os.path.exists(destination):
            print(f"Downloading {filename}...")
            try:
                # Download the file from Hugging Face Hub
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="model"
                )
                # Create destination directory if needed
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                # Copy or move the file to the destination
                import shutil
                shutil.copy(file_path, destination)
                print(f"Downloaded {filename} to {destination}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"File {destination} already exists, skipping download")
        
        downloaded_paths[filename] = destination
    
    # Download representative images or data folder if needed
    # If you have a separate repository for images or data
    # You can use snapshot_download to get the whole directory
    
    # For example, if you have a data repository:
    # data_repo_id = "YOUR_USERNAME/plant-data"
    # data_path = snapshot_download(repo_id=data_repo_id, repo_type="dataset", local_dir="data")
    
    print("All model files downloaded or already exist.")
    return downloaded_paths

# Then call this function in your startup event
@app.on_event("startup")
async def load_models():
    """Load all ML models on server startup"""
    global classification_pipeline, llm_qa, retrieval_system, plant_metadata
    
    # First download the model files
    downloaded_paths = download_model_files()
    
    # Update model paths if needed
    MODEL_CONFIG["classifier_path"] = downloaded_paths.get("dino_best.pth", MODEL_CONFIG["classifier_path"])
    MODEL_CONFIG["embed_path"] = downloaded_paths.get("embedding_best.pth", MODEL_CONFIG["embed_path"])
    MODEL_CONFIG["retrieval_index_file"] = downloaded_paths.get("plant_index_2.pkl", MODEL_CONFIG["retrieval_index_file"])
    MODEL_CONFIG["metadata_path"] = downloaded_paths.get("merge_metadata.json", MODEL_CONFIG["metadata_path"])
    
    print("Initializing Classification Pipeline...")
    classification_pipeline = SimpleClassificationPipeline(
        classifier_path=MODEL_CONFIG["classifier_path"],
        label_path=MODEL_CONFIG["labels_path"]
    )
    
    print("Initializing LLM...")
    llm_qa = PlantQA(metadata_path=MODEL_CONFIG["metadata_path"])
    
    print("Initializing Retrieval System...")
    retrieval_system = ImageRetrievalSystem(
        model_path=MODEL_CONFIG["embed_path"],
        image_folder=MODEL_CONFIG["labels_path"],
        index_file=MODEL_CONFIG["retrieval_index_file"]
    )
    
    print("All models loaded successfully!")
    print("Loading Plant Metadata...")
    try:
        with open(MODEL_CONFIG["metadata_path"], 'r', encoding='utf-8') as f:
            plant_metadata = json.load(f)
        print(f"Loaded metadata for {len(plant_metadata)} plant species")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        plant_metadata = {}


# Pydantic models for request/response
class ClassificationResult(BaseModel):
    label: str
    confidence: float
    image_path: Optional[str] = None

class ClassificationResponse(BaseModel):
    results: List[ClassificationResult]

class QARequest(BaseModel):
    label: str
    question: str

class QAResponse(BaseModel):
    answer: str


@app.on_event("shutdown")
async def cleanup():
    """Clean up resources when shutting down"""
    global classification_pipeline, llm_qa, retrieval_system
    
    print("Cleaning up resources...")
    if classification_pipeline:
        classification_pipeline.cleanup()
    if llm_qa:
        llm_qa.close()
    
    # Clean temporary files
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Cleanup complete!")

# Add or replace this endpoint in your main.py file

@app.get("/api/plants")
async def get_plants():
    """Return all plants metadata for the library mode"""
    print("Received request for /api/plants")
    
    if not plant_metadata:
        print("Error: Plant metadata not loaded")
        raise HTTPException(
            status_code=500, 
            detail="Plant metadata not loaded. Please check the server configuration."
        )
    
    try:
        # Check data integrity
        if not isinstance(plant_metadata, dict):
            print(f"Error: Unexpected metadata type: {type(plant_metadata)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected metadata format: {type(plant_metadata)}"
            )
        
        metadata_count = len(plant_metadata)
        print(f"Successfully serving metadata for {metadata_count} plants")
        
        # You might want to limit the data if it's too large
        # For now, we're sending everything
        return plant_metadata
        
    except Exception as e:
        print(f"Error in /api/plants endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing plant metadata: {str(e)}"
        )

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_image(file: UploadFile = File(...)):
    """Classify an uploaded plant image and return top 5 predictions"""
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Open image and process through retrieval system first
        pil_image = Image.open(file_path).convert("RGB")
        _, _, distances = retrieval_system.retrieve_similar_images(pil_image, top_k=5)
        min_distance = np.min(distances)
        
        # Check if image is too dissimilar from known plants
        if min_distance > 0.25:
            return ClassificationResponse(results=[
                ClassificationResult(
                    label="Không tồn tại trong cơ sở dữ liệu",
                    confidence=1.0,
                    image_path=None
                )
            ])
        
        # Get top 5 classifications
        top5 = classification_pipeline.process_image_topk(file_path, top_k=5)
        
        # Format results
        results = []
        for label, prob, rep_img in top5:
            results.append(
                ClassificationResult(
                    label=label,
                    confidence=float(prob),
                    image_path=rep_img if rep_img else None
                )
            )
        
        return ClassificationResponse(results=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer a question about a specific plant"""
    if not request.label or not request.question:
        raise HTTPException(status_code=400, detail="Both label and question are required")
    
    try:
        # Extract plain label from label string if in format "Label (99.99%)"
        import re
        match = re.match(r"^(.+?)\s\(\d+\.\d+%\)$", request.label)
        label = match.group(1) if match else request.label
        
        # Generate answer
        answer = llm_qa.generate_answer(label, request.question)
        return QAResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

# Serve static files (React frontend)
app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)