import os
import torch
import numpy as np
import faiss
from PIL import Image
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from torchvision import transforms

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.convnext = model.convnext
        self.vit = model.vit
        self.fusion = model.fusion
        
    def forward(self, x):
        # ConvNeXt feature extraction
        conv_features = self.convnext(x)[-1]
        
        # Vision Transformer feature extraction
        vit_features = self.vit(pixel_values=x).last_hidden_state
        B, N, D = vit_features.shape
        
        # Remove the [CLS] token and reshape
        vit_features = vit_features[:, 1:, :]
        H, W = conv_features.shape[2], conv_features.shape[3]
        
        # Reshape ViT output to match ConvNeXt spatial dimensions
        vit_features = vit_features.permute(0, 2, 1)
        vit_features = vit_features.reshape(B, D, int((N-1)**0.5), int((N-1)**0.5))
        
        # Resize ViT features to match ConvNeXt spatial dimensions
        vit_features = F.interpolate(vit_features, size=(H, W), mode='bilinear', align_corners=False)
        
        # Extract fused features
        fused_features = self.fusion(conv_features, vit_features)
        
        return fused_features

class ImageRetrievalSystem:
    def __init__(self, model_path, image_folder, index_file="plant_index.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load trained model
        self.model = self.load_model(model_path)
        self.feature_extractor = FeatureExtractor(self.model).to(self.device)
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize FAISS index and image database
        self.image_folder = image_folder
        self.index_file = index_file
        self.image_paths = []
        self.image_labels = []
        self.index = None
        
        # Build or load the image database and FAISS index
        if os.path.exists(self.index_file):
            self.load_index()
        else:
            self.build_index()
        
    def load_model(self, model_path):
        # Determine the number of classes from the model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        # This assumes ImageEmbeddingModel is imported from elsewhere
        from embedder import ImageEmbeddingModel
        model = ImageEmbeddingModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def extract_features(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
        features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()
    
    def build_index(self):
        print("Building FAISS index...")
        features_list = []
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    self.image_paths.append(full_path)
                    self.image_labels.append(label)
        
        for img_path in tqdm(self.image_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                features = self.extract_features(img)
                features_list.append(features.flatten())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                idx = self.image_paths.index(img_path)
                self.image_paths.pop(idx)
                self.image_labels.pop(idx)
        
        if features_list:
            all_features = np.vstack(features_list)
            d = all_features.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(all_features)
            self.save_index()
            print(f"FAISS index built with {len(self.image_paths)} images.")
        else:
            print("No valid images found.")
    
    def save_index(self):
        print(f"Saving index to {self.index_file}")
        with open(self.index_file, 'wb') as f:
            pickle.dump({
                'image_paths': self.image_paths,
                'image_labels': self.image_labels,
                'index_data': faiss.serialize_index(self.index)
            }, f)
    
    def load_index(self):
        print(f"Loading index from {self.index_file}")
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
        self.image_paths = data['image_paths']
        self.image_labels = data['image_labels']
        self.index = faiss.deserialize_index(data['index_data'])
        print(f"FAISS index loaded with {len(self.image_paths)} images.")
    
    def retrieve_similar_images(self, query_img, top_k=5):
        query_features = self.extract_features(query_img)
        D, I = self.index.search(query_features, top_k)
        similar_paths = [self.image_paths[i] for i in I[0]]
        similar_labels = [self.image_labels[i] for i in I[0]]
        return similar_paths, similar_labels, D[0]