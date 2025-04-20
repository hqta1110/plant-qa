from PIL import Image
import torch
from torchvision import transforms
import json
import gc
from classifier import HybridClassifier
from torchvision import datasets
import os
class SimpleClassificationPipeline:
    def __init__(self, classifier_path=None, label_path=None):
        print("Loading classifier...")
        self.classifier = HybridClassifier()
        if classifier_path:
            checkpoint = torch.load(classifier_path, map_location='cuda', weights_only=True)
            self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        print("Loaded classifier")
        
        # Define the image transformation.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.labels_path = label_path
        # Load labels if provided.
        if self.labels_path:
            # self.labels = json.load(open(label_path, 'r'))
            self.labels = datasets.ImageFolder(root=label_path).classes
        else:
            self.labels = None
    def get_representative_image(self, label):
        """
        Returns a representative image path for a given label.
        Assumes that your `labels_path` directory contains subfolders named after labels.
        """
        label_folder = os.path.join(self.labels_path, label)
        if os.path.isdir(label_folder):
            for f in os.listdir(label_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    return os.path.join(label_folder, f)
        # Fallback: if no image is found, return None or a default image.
        return None

    def process_image(self, image_path):
        # Load the image and apply transformations.
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()
            self.classifier.cuda()
        
        # Run the classifier.
        with torch.no_grad():
            logits = self.classifier(tensor_image)
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
        
        # Get the predicted label.
        if self.labels and pred_idx.item() < len(self.labels):
            pred_label = self.labels[pred_idx.item()]
        else:
            pred_label = f"Class {pred_idx.item()}"
        
        print(f"Predicted label: {pred_label}, Confidence: {max_prob.item():.4f}")
        return pred_label, max_prob.item()
    
    def process_image_topk(self, image_path, top_k=5):
        # Load the image and apply transformations.
        image = Image.open(image_path).convert("RGB")
        tensor_image = self.transform(image).unsqueeze(0)  # Add batch dimension

        if torch.cuda.is_available():
            tensor_image = tensor_image.cuda()
            self.classifier.cuda()

        # Run the classifier.
        with torch.no_grad():
            logits = self.classifier(tensor_image)
            probs = torch.softmax(logits, dim=1)
            topk_probs, topk_indices = torch.topk(probs, top_k)

        results = []
        for idx, prob in zip(topk_indices[0].tolist(), topk_probs[0].tolist()):
            # Get the predicted label.
            if self.labels and idx < len(self.labels):
                label = self.labels[idx]
            else:
                label = f"Class {idx}"
            # Get a representative image for this label.
            rep_img = self.get_representative_image(label)
            results.append((label, prob, rep_img))
        return results
    
    

    def cleanup(self):
        print("Cleaning up resources...")
        del self.classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Cleanup complete.")
