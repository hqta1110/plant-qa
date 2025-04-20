import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torchvision import datasets
from huggingface_hub import login
import unicodedata
class PlantQA:
    def __init__(self,
                 metadata_path,
                 tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
                 llm_model="Qwen/Qwen2.5-7B-Instruct",
                 max_length=2048,
                 device='cuda'):
        """
        Initialize the PlantQA module using Hugging Face transformers.
        Args:
            metadata_path (str): Path to a JSON file mapping plant labels to metadata.
            tokenizer_name (str): The name or path of the tokenizer.
            llm_model (str): The name or path of the LLM model.
            max_length (int): Maximum length for generated sequences.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        # Load plant metadata from JSON
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        print("Loaded metadata")

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize the model
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Store generation parameters
        self.generation_config = {
            "max_length": max_length,
            "temperature": 0.5,
            "top_p": 0.8,
            "repetition_penalty": 1.05,
            "do_sample": True
        }
    def normalize_name(self,name):
        """
        Normalize names by:
        - Converting to lowercase
        - Stripping leading/trailing whitespace
        - Removing invisible Unicode characters
        - Replacing multiple spaces with a single space
        """
        name = unicodedata.normalize("NFKC", name)  # Normalize Unicode characters
        name = name.lower().strip()  # Convert to lowercase and strip spaces
        name = " ".join(name.split())  # Replace multiple spaces with a single space
        return name
    def search_normalized_name(self, json_data, name):
        """
        Search for a name in a JSON list after normalizing both.
        
        :param json_data: List of names from JSON
        :param name: Name to search for
        :return: Boolean indicating if the name is found
        """
        print(type(json_data))
        normalized_name = self.normalize_name(name)
        for entry in json_data.keys():
            if normalized_name == self.normalize_name(entry):
                return json_data[entry]
        return None
    def generate_answer(self, label, question):
        """
        Generate an answer to the user's question using the plant metadata as context.
        Args:
            label (str): The plant label from the image classification module.
            question (str): The user's question.
        Returns:
            str: The generated answer from the LLM.
        """
        # Retrieve metadata corresponding to the given label
        # plant_metadata = self.metadata.get(label, "")
        plant_metadata = self.search_normalized_name(self.metadata, label)
        
        if plant_metadata:
            print("Retrieving data")
            system_message = (
                "Bạn là một trợ lý ảo tiếng Việt chuyên về thực vật học. "
                f"Dưới đây là một số thông tin liên quan đến loại cây này:\n{plant_metadata}"
            )
        else:
            print("Unseen plant")
            system_message = (
                "Bạn là một trợ lý ảo tiếng Việt chuyên về thực vật học. "
                "Không có thông tin metadata cụ thể cho loại cây này."
            )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]

        # Create the prompt using the chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Print prompt length for debugging
        print(f"Prompt length: {len(prompt_text)}")
        print("Prompt text:")
        print(prompt_text)
        print("-" * 50)

        # Tokenize the input
        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_length = inputs['input_ids'].shape[1]
        print(f"Input token length: {input_length}")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generation_config,
                min_length=input_length + 1  # Ensure we generate at least one new token
            )

        # Get the length of the output
        output_length = outputs.shape[1]
        print(f"Output token length: {output_length}")

        # Decode only the new tokens (response)
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Print full generated text for debugging
        full_generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Full generated text:")
        print(full_generated)
        print(f"Full generated length: {len(full_generated)}")
        print("-" * 50)
        print("Extracted response:")
        print(response)
        print(f"Response length: {len(response)}")
        
        return response

    def convert(self, value):
        with open('/home/sora/code/name_mapping.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            for key, val in metadata.items():
                if val == value:
                    return key
        return None

    def close(self):
        """
        Clean up resources.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()