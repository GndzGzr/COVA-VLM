from typing import Union, Optional, List, Dict, Any
from PIL import Image
import torch
import os


try:
    from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
except ImportError:
    raise ImportError(
        "Could not import LlavaForConditionalGeneration from transformers. "
        "Please install the latest version:\n"
        "pip install --upgrade transformers>=4.37.0\n"
        "pip install --upgrade accelerate\n"
        "pip install protobuf<=3.20.0\n"
        "You might also need: pip install bitsandbytes"
    )

class VisionLanguageModel:
    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        """
        Initialize Vision-Language model (LLaVA)
        Args:
            model_path: Path to the model or model name in HuggingFace hub
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(self.device)
        
        # Load processor and model with specific configuration
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=False,  # Use slow tokenizer for better compatibility
        )
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if self.device == "cpu":
            self.model.to(self.device)

    def generate_response(
        self, 
        image: Union[Image.Image, str], 
        question: str,
        objects: List[Dict[str, Any]] = None,
        mode: str = "chat",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response for the given image and question
        Args:
            image: PIL Image or path to image file
            question: Question text
            objects: List of detected objects with distances and danger levels (for walking assistance mode)
            mode: "chat" or "walking assistance"
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        Returns:
            str: Generated response
        """
        # Load image if path is provided
        if isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                raise ValueError(f"Error loading image: {str(e)}")
        
        # Prepare inputs with explicit handling
        try:
            # Format the prompt based on mode
            if mode == "chat":
                # Base prompt for regular chat mode
                base_prompt = "You are a helpful visual assistant that can see and understand images. Answer the following question about the image."
                formatted_prompt = f"Human: <image>\n{base_prompt}\n{question}\n\nAssistant: "
            
            elif mode == "walking assistance":
                # Base prompt for walking assistance mode
                base_prompt = "You are a visual assistant model for a visually impaired person. Please analyze the image and generate a short spoken instruction, considering object types, their distances, and whether they are dangerous. Give concise and helpful guidance like \"turn left\", \"stop\", or \"go ahead\".\n\nPlease provide a short voice-assistant style instruction to the user based on the information above."
                
                # Format objects information
                objects_info = ""
                if objects:
                    objects_info = "Detected objects:\n"
                    for obj in objects:
                        objects_info += f"- {obj['Object']} at {obj['Distance']} meters, danger level: {obj['Danger']}\n"
                
                formatted_prompt = f"Human: <image>\n{base_prompt}\n\n{objects_info}\n{question}\n\nAssistant: "
            else:
                formatted_prompt = f"Human: <image>\n{question}\n\nAssistant: "
            
            # Process inputs
            inputs = self.processor(
                images=image,
                text=formatted_prompt,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=2048  # Maximum context length for the model
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                    max_new_tokens=max_new_tokens,  # Control length of new generated tokens
                    min_new_tokens=10,  # Ensure some minimal response
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response
            if "Assistant:" in response:
                response = response.split("Assistant:", 1)[1]
            if "Human:" in response:
                response = response.split("Human:", 1)[0]
                
            return response.strip()
            
        except Exception as e:
            raise ValueError(f"Error processing image or generating response: {str(e)}")

    def __call__(self, image: Union[Image.Image, str], question: str, objects: List[Dict[str, Any]] = None, mode: str = "chat") -> str:
        """
        Convenience method to call generate_response
        """
        return self.generate_response(image, question, objects, mode) 