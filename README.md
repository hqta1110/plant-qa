# Plant Classification and Q&A System

This application allows users to:
1. Upload images of plants for classification
2. Browse a plant library for information
3. Ask questions about specific plants

## Features

- **Plant Classification**: Upload images to identify plant species
- **Plant Library**: Browse a comprehensive database of plant information
- **Question Answering**: Ask specific questions about plants and get AI-generated responses

## Deployment on Hugging Face Spaces

This application is configured for deployment on Hugging Face Spaces using Docker.

### Repository Structure

```
├── main.py                # FastAPI backend
├── classifier.py          # Plant classification model
├── embedder.py            # Image embedding model
├── llm_huggingface.py     # Plant QA system
├── retrieval_system.py    # Image retrieval system
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── frontend/             # React frontend
```

### Model Files

This application references the following model files, which are automatically downloaded during startup:
- `dino_best.pth`: Classification model
- `embedding_best.pth`: Image embedding model
- `plant_index_2.pkl`: Retrieval system index
- `merge_metadata.json`: Plant metadata

These files are sourced from the Hugging Face repository: `hqta1110/plant-classification-models`

## Local Development

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up the frontend:
   ```
   cd frontend
   npm install
   npm run build
   ```

3. Run the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

## Usage

1. **Classification**: Upload a plant image to identify its species
2. **Library Mode**: Browse the catalog of plants and access detailed information
3. **Q&A**: Ask questions about specific plants and receive detailed answers

## License

[Specify your license information here]