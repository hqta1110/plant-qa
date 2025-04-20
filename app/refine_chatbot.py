import os
import torch
import numpy as np
import faiss
from PIL import Image
import gradio as gr
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
import pickle
import gc
import tempfile
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Import your model and pipelines
from classifier import HybridClassifier  # used for retrieval feature extraction
from simImage import SimpleClassificationPipeline  # classification pipeline
from llm_huggingface import PlantQA  # LLM for Q&A
from embedder import ImageEmbeddingModel
# from huggingface_hub import login
# login('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')

###############################
# Retrieval Module Components #
###############################
# (Your FeatureExtractor and ImageRetrievalSystem classes remain unchanged.)
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

##########################################
# Chatbot App with Two-Step Interaction  #
##########################################
class PlantQAInterface:
    def __init__(self, cls_path, embed_path, labels_path, metadata_path, retrieval_index_file="plant_index.pkl"):
        self.cls_path = cls_path
        self.embed_path = embed_path
        self.labels_path = labels_path  # also used as image folder for retrieval
        self.metadata_path = metadata_path
        self.text_prompts = [["tree."], ["plant."], ["flower."], ["leaf."]]
        
        # Initialize the classification pipeline
        print("Initializing Classification Pipeline...")
        self.pipeline = SimpleClassificationPipeline(
            classifier_path=self.cls_path,
            label_path=self.labels_path
        )
        
        # Initialize the LLM
        print("Initializing LLM...")
        self.llm = PlantQA(metadata_path=self.metadata_path)
        
        # Initialize the retrieval system
        print("Initializing Retrieval System...")
        self.retrieval_system = ImageRetrievalSystem(
            model_path=self.embed_path,
            image_folder=self.labels_path,
            index_file=retrieval_index_file
        )
        
        print("All models loaded successfully!")
    
    def get_top5_labels(self, image):
        # If image is a file path, open it; otherwise, assume it is a PIL image.
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
            image_path = image
        else:
            pil_image = image
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            pil_image.save(temp_file.name)
            image_path = temp_file.name

        # Run retrieval to check similarity (optional, as in your original logic)
        _, _, distances = self.retrieval_system.retrieve_similar_images(pil_image, top_k=5)
        min_distance = np.min(distances)
        print(f"Minimum retrieval distance: {min_distance:.4f}")

        # If retrieval indicates the image is too far from known plants, return a default message.
        if min_distance > 0.25:
            return [("Không tồn tại trong cơ sở dữ liệu", 1.0)]
        else:
            # Here we assume your pipeline now supports a top-k mode.
            # For example, process_image_topk returns a list of (label, probability).
            top5 = self.pipeline.process_image_topk(image_path, top_k=5)
            return top5
    
    def generate_answer(self, selected_label, question):
        import re
        # Generate answer using the LLM given the selected label and question.
        if not selected_label:
            return "Vui lòng chọn một loại cây trước khi đặt câu hỏi."
        match = re.match(r"^(.+?)\s\(\d+\.\d+%\)$", selected_label)
        if match:
            return self.llm.generate_answer(match.group(1), question)
        return text
    
    def launch_interface(self):
        # Using Gradio Blocks to create a two-step interface.
        with gr.Blocks() as demo:
            gr.Markdown("# Plant Classification and Q&A System")

            with gr.Tab("Step 1: Classification"):
                with gr.Row():
                    image_input = gr.Image(type="filepath", label="Upload Plant Image")
                    btn_classify = gr.Button("Classify")
                # Create the gallery without using .style()
                gallery = gr.Gallery(label="Top 5 Predictions")
                # A radio component to let the user choose an option.
                selected_option = gr.Radio(choices=[], label="Select the best match")


            with gr.Tab("Step 2: Q&A"):
                question_input = gr.Textbox(label="Ask a question about the plant", lines=2)
                btn_answer = gr.Button("Get Answer")
                answer_output = gr.Textbox(label="Response", lines=10)

            # Step 1: Classification function.
            def classify_image(image):
                if image is None:
                    return gr.update(value=[]), gr.update(choices=["Please upload an image."])
                top5 = self.get_top5_labels(image)
                gallery_items = []
                radio_options = []
                for label, prob, rep_img in top5:
                    caption = f"{label} ({prob*100:.2f}%)"
                    # If a representative image exists, include it; otherwise, use a placeholder.
                    gallery_items.append((rep_img if rep_img is not None else "", caption))
                    radio_options.append(caption)
                return gallery_items, gr.update(value=None, choices=radio_options)

            btn_classify.click(fn=classify_image, inputs=image_input, outputs=[gallery, selected_option])

            # Step 2: Q&A function.
            btn_answer.click(fn=self.generate_answer, inputs=[selected_option, question_input], outputs=answer_output)

        return demo
    
    def cleanup(self):
        print("Cleaning up resources...")
        if hasattr(self, 'pipeline'):
            self.pipeline.cleanup()
            del self.pipeline
        if hasattr(self, 'llm'):
            self.llm.close()
            del self.llm
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup complete!")

def main():
    # Configuration paths
    ckpt_path = '/home/sora/anhhqt/tmp/dino_best.pth'
    embed_path = '/home/sora/anhhqt/tmp/embedding_best.pth'
    labels_path = '/home/sora/pretrain-llm/data/data'
    metadata_path = '/home/sora/pretrain-llm/code/merge_metadata.json'
    retrieval_index_file = '/home/sora/pretrain-llm/llm-va/plant_index_2.pkl'
    
    try:
        plant_qa = PlantQAInterface(ckpt_path, embed_path, labels_path, metadata_path, retrieval_index_file)
        interface = plant_qa.launch_interface()
        interface.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            # Optionally add cleanup on server shutdown
            # _cleanup=lambda: plant_qa.cleanup()
        )
    except Exception as e:
        print(f"An error occurred during startup: {str(e)}")
        if 'plant_qa' in locals():
            plant_qa.cleanup()

if __name__ == "__main__":
    main()
