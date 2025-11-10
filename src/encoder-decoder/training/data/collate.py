"""Collate function for batch processing"""

from typing import Dict, List
import torch


def make_collate(tokenizer, max_ans_toks: int, system_prompt: str = ""):
    """
    Create collate function for DataLoader.
    
    Args:
        tokenizer: Hugging Face tokenizer
        max_ans_toks: Maximum answer tokens
        system_prompt: System prompt to use in chat template
        
    Returns:
        Collate function that processes batch items
    """
    # Use default if not provided
    if not system_prompt:
        system_prompt = "You are an expert autonomous driving assistant. Analyze the 3D LiDAR point cloud and camera images to understand the driving scene. Provide accurate, concise descriptions of objects, their locations, distances, and spatial relationships. Use directional terms like 'ahead', 'left', 'right', 'behind' and specify distances in meters when describing object locations."
    
    def collate(items: List[Dict]):
        bevs = [it["bev"] for it in items]
        tokens = [it["token"] for it in items]
        questions = [it["question"] for it in items]
        answers = [it["answer"] for it in items]
        
        prompts = []
        for q in questions:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            )
            
        prompt_batch = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
        ans_batch = tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_ans_toks,
            add_special_tokens=True,
        )
        
        return {
            "bev": torch.stack(bevs, dim=0),
            "sample_tokens": tokens,
            "prompt_ids": prompt_batch["input_ids"],
            "prompt_attn": prompt_batch["attention_mask"],
            "answer_ids": ans_batch["input_ids"],
            "answer_attn": ans_batch["attention_mask"],
        }
        
    return collate
