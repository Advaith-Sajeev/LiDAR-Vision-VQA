"""
Inference engine for LiDAR-Vision-LLM
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Union
from pathlib import Path


class InferenceEngine:
    """
    High-level inference engine for LiDAR-Vision-LLM.
    
    Handles:
      - Prompt formatting
      - Feature processing
      - Model forward pass
      - Text generation
    """
    
    def __init__(self, models: Dict):
        """
        Initialize inference engine.
        
        Args:
            models: Dictionary of models from ModelLoader.load_all()
        """
        self.tokenizer = models["tokenizer"]
        self.base_model = models["base_model"]
        self.vat_lidar = models["vat_lidar"]
        self.vat_vision = models.get("vat_vision")
        self.vision_adapter = models.get("vision_adapter")
        self.runtime = models.get("runtime")
        self.nusc = models.get("nusc")
        self.config = models["config"]
        self.device = models["device"]
        self.d_model = models["d_model"]
        
        self.use_vision = self.config.get("use_vision", False) and self.vat_vision is not None
        self.prefix_scale = self.config.get("prefix_scale", 0.2)
        
        # Special token IDs
        self.lidar_start_id = self.tokenizer.convert_tokens_to_ids("<lidar_start>")
        self.lidar_end_id = self.tokenizer.convert_tokens_to_ids("<lidar_end>")
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<vision_start>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<vision_end>")
        
        print("[engine] Inference engine initialized")
        print(f"[engine] Vision pipeline: {'enabled' if self.use_vision else 'disabled'}")
    
    def format_prompt(self, question: str, include_vision: bool = True) -> str:
        """
        Format input prompt with special tokens.
        
        Args:
            question: User question
            include_vision: Whether to include vision tokens
            
        Returns:
            Formatted prompt string
        """
        if self.use_vision and include_vision:
            return f"<vision_start><vision_end><lidar_start><lidar_end>{question}\nAnswer:"
        else:
            return f"<lidar_start><lidar_end>{question}\nAnswer:"
    
    def process_lidar(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Process BEV LiDAR features through VAT.
        
        Args:
            bev: BEV features [B, C, H, W] or [C, H, W]
            
        Returns:
            LiDAR prompts [B, n_queries, d_model]
        """
        if bev.ndim == 3:
            bev = bev.unsqueeze(0)  # Add batch dimension
        
        bev = bev.to(self.device)
        
        with torch.no_grad():
            lidar_prompts = self.vat_lidar(bev)  # [B, n_queries, d_model]
        
        return lidar_prompts
    
    def process_vision(self, sample_token: str) -> Optional[torch.Tensor]:
        """
        Process multi-view camera images through DeepEncoder and Vision VAT.
        
        Args:
            sample_token: nuScenes sample token
            
        Returns:
            Vision prompts [B, n_queries, d_model] or None if vision disabled
        """
        if not self.use_vision:
            return None
        
        with torch.no_grad():
            # Get camera images for this sample
            sample = self.nusc.get("sample", sample_token)
            cam_tokens = [sample["data"][cam] for cam in [
                "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK", "CAM_BACK_RIGHT", "CAM_BACK_LEFT"
            ]]
            
            # Process through DeepEncoder
            kv_tokens = self.runtime.process_sample_token(
                sample_token=sample_token,
                nusc=self.nusc
            )  # [N_img_tokens, 1024]
            
            # Project to d_model
            kv_tokens = self.runtime.projector(kv_tokens)  # [N_img_tokens, 2048]
            kv_tokens = self.vision_adapter(kv_tokens.unsqueeze(0))  # [1, N_img_tokens, d_model]
            
            # Process through Vision VAT
            vision_prompts = self.vat_vision(kv_tokens)  # [1, n_queries, d_model]
        
        return vision_prompts
    
    def build_inputs_embeds(
        self,
        prompt: str,
        lidar_prompts: torch.Tensor,
        vision_prompts: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Build inputs_embeds by interleaving text and modal prompts.
        
        Args:
            prompt: Formatted prompt string
            lidar_prompts: LiDAR prompts [B, n_queries, d_model]
            vision_prompts: Vision prompts [B, n_queries, d_model] or None
            
        Returns:
            Tuple of (inputs_embeds, attention_mask)
        """
        # Tokenize
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.device)  # [1, L]
        
        # Get text embeddings
        text_embeds = self.base_model.get_input_embeddings()(input_ids)  # [1, L, d_model]
        
        # Find special token positions
        ids_flat = input_ids[0]
        
        # Build sequence
        embeds_list = []
        pos = 0
        
        # Handle vision tokens
        if self.use_vision and vision_prompts is not None:
            vision_start_pos = (ids_flat == self.vision_start_id).nonzero(as_tuple=True)[0]
            vision_end_pos = (ids_flat == self.vision_end_id).nonzero(as_tuple=True)[0]
            
            if len(vision_start_pos) > 0 and len(vision_end_pos) > 0:
                vs = vision_start_pos[0].item()
                ve = vision_end_pos[0].item()
                
                # Text before vision
                if vs > pos:
                    embeds_list.append(text_embeds[:, pos:vs, :])
                
                # Vision start token
                embeds_list.append(text_embeds[:, vs:vs+1, :])
                
                # Vision prompts (scaled)
                embeds_list.append(vision_prompts * self.prefix_scale)
                
                # Vision end token
                embeds_list.append(text_embeds[:, ve:ve+1, :])
                
                pos = ve + 1
        
        # Handle LiDAR tokens
        lidar_start_pos = (ids_flat == self.lidar_start_id).nonzero(as_tuple=True)[0]
        lidar_end_pos = (ids_flat == self.lidar_end_id).nonzero(as_tuple=True)[0]
        
        if len(lidar_start_pos) > 0 and len(lidar_end_pos) > 0:
            ls = lidar_start_pos[0].item()
            le = lidar_end_pos[0].item()
            
            # Text before lidar
            if ls > pos:
                embeds_list.append(text_embeds[:, pos:ls, :])
            
            # LiDAR start token
            embeds_list.append(text_embeds[:, ls:ls+1, :])
            
            # LiDAR prompts (scaled)
            embeds_list.append(lidar_prompts * self.prefix_scale)
            
            # LiDAR end token
            embeds_list.append(text_embeds[:, le:le+1, :])
            
            pos = le + 1
        
        # Remaining text
        if pos < text_embeds.shape[1]:
            embeds_list.append(text_embeds[:, pos:, :])
        
        # Concatenate
        inputs_embeds = torch.cat(embeds_list, dim=1)  # [1, total_len, d_model]
        
        # Create attention mask
        attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=self.device)
        
        return inputs_embeds, attention_mask
    
    @torch.no_grad()
    def generate(
        self,
        question: str,
        bev: Union[torch.Tensor, np.ndarray, str, Path],
        sample_token: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> str:
        """
        Generate answer for a question given LiDAR (and optionally vision) data.
        
        Args:
            question: User question
            bev: BEV features as tensor, array, or path to .npy file
            sample_token: nuScenes sample token (required for vision)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            do_sample: Whether to sample (vs greedy decode)
            num_beams: Number of beams for beam search
            
        Returns:
            Generated answer string
        """
        # Load BEV if path provided
        if isinstance(bev, (str, Path)):
            bev = np.load(bev)
        if isinstance(bev, np.ndarray):
            bev = torch.from_numpy(bev).float()
        
        # Process LiDAR
        lidar_prompts = self.process_lidar(bev)
        
        # Process vision (if enabled and sample_token provided)
        vision_prompts = None
        include_vision = False
        if self.use_vision and sample_token is not None:
            try:
                vision_prompts = self.process_vision(sample_token)
                include_vision = True
            except Exception as e:
                print(f"[engine] Warning: Failed to process vision: {e}")
        
        # Format prompt
        prompt = self.format_prompt(question, include_vision=include_vision)
        
        # Build inputs
        inputs_embeds, attention_mask = self.build_inputs_embeds(
            prompt, lidar_prompts, vision_prompts
        )
        
        # Generate
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 50,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode (skip prompt tokens)
        generated_ids = outputs[0][inputs_embeds.shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer.strip()
    
    @torch.no_grad()
    def generate_batch(
        self,
        questions: List[str],
        bevs: List[Union[torch.Tensor, np.ndarray, str, Path]],
        sample_tokens: Optional[List[str]] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate answers for a batch of questions.
        
        Args:
            questions: List of questions
            bevs: List of BEV features
            sample_tokens: List of sample tokens (optional, for vision)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated answers
        """
        if sample_tokens is None:
            sample_tokens = [None] * len(questions)
        
        answers = []
        for q, bev, token in zip(questions, bevs, sample_tokens):
            answer = self.generate(q, bev, token, **generation_kwargs)
            answers.append(answer)
        
        return answers
