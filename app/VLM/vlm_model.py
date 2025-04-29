from typing import Union, Optional, List, Dict, Any
from PIL import Image
import torch
import os
import gc  # Add garbage collector


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
    _model_instance = None
    _processor_instance = None
    
    def __init__(self, model_path: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        """
        Initialize Vision-Language model (LLaVA)
        Args:
            model_path: Path to the model or model name in HuggingFace hub
            device: Device to run the model on ('cuda' or 'cpu')
        """
        # Force CPU usage for the LLaVA model to prevent OOM errors
        self.device = "cpu"  # Always use CPU for this large model
        print(f"Using device: {self.device} for VLM model")
        
        # Use cached instances if available to avoid reloading
        if VisionLanguageModel._processor_instance is None:
            print("Loading VLM processor for the first time...")
            VisionLanguageModel._processor_instance = AutoProcessor.from_pretrained(
                model_path,
                use_fast=False,  # Use slow tokenizer for better compatibility
            )
        
        if VisionLanguageModel._model_instance is None:
            print("Loading VLM model for the first time...")
            # Configure model for optimal performance and lower memory usage
            torch_dtype = torch.float32  # Use float32 for CPU
            
            VisionLanguageModel._model_instance = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=None,  # Don't use auto device mapping
                low_cpu_mem_usage=True,
                offload_folder="offload",  # Enable model offloading
                offload_state_dict=True,   # Offload state dict
            )
            
            # Move to CPU
            VisionLanguageModel._model_instance.to(self.device)
                
            # Optimize for inference
            VisionLanguageModel._model_instance.eval()
        
        self.processor = VisionLanguageModel._processor_instance
        self.model = VisionLanguageModel._model_instance
        
        # Warm up the model with a small dummy input
        self._warmup()
        
    def _warmup(self):
        """Warm up the model with a dummy image and question"""
        try:
            # Only warm up on first initialization
            if not hasattr(VisionLanguageModel, '_warmed_up'):
                print("Warming up VLM model...")
                dummy_image = Image.new('RGB', (224, 224), color='white')
                dummy_question = "What's in this image?"
                
                inputs = self.processor(
                    images=dummy_image,
                    text=f"Human: <image>\n{dummy_question}\n\nAssistant: ",
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    self.model.generate(**inputs, max_new_tokens=5)  # Reduced tokens for warmup
                
                VisionLanguageModel._warmed_up = True
                print("VLM model warm-up complete")
                
                # Clean up after warmup
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Warning: VLM warm-up failed: {str(e)}")

    def generate_response(
        self, 
        image: Union[Image.Image, str], 
        question: str,
        objects: List[Dict[str, Any]] = None,
        mode: str = "chat",
        max_new_tokens: int = 256,  # Reduced token count
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
        
        # Convert CV2 image to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
        
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
                max_length=1024  # Reduced context length from 2048 to 1024
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
                    max_new_tokens=max_new_tokens,  # Reduced from 512 to 256
                    min_new_tokens=10,  # Ensure some minimal response
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            print(response)
            
            # Clean up the response
            if "Assistant:" in response:
                response = response.split("Assistant:", 1)[1]
            if "Human:" in response:
                response = response.split("Human:", 1)[0]
                
            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            return response.strip()
            
        except Exception as e:
            # Clean up on exception
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            raise ValueError(f"Error processing image or generating response: {str(e)}")

    def __call__(self, image: Union[Image.Image, str], question: str, objects: List[Dict[str, Any]] = None, mode: str = "chat") -> str:
        """
        Convenience method to call generate_response
        """
        return self.generate_response(image, question, objects, mode) 