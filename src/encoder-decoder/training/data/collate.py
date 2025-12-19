"""Collate function for batch processing"""

from typing import Dict, List, Optional
import torch


def make_collate(tokenizer, max_ans_toks: int, system_prompt: str = "", load_images: bool = False):
    """
    Create collate function for DataLoader.
    
    Args:
        tokenizer: Hugging Face tokenizer
        max_ans_toks: Maximum answer tokens
        system_prompt: System prompt to use in chat template
        load_images: Whether to expect and collate camera images
        
    Returns:
        Collate function that processes batch items
    """
    # Use default if not provided
    if not system_prompt:
        system_prompt = """You are an Autonomous Driving Perception Assistant analyzing six surround-view camera images (Front, Front-Left, Front-Right, Back, Back-Left, Back-Right). Your task is to fuse these views into a consistent scene understanding.

Follow the output template based on the requested mode:

1. @SCENE_SUMMARY_MODE
   Output Template: A cohesive narrative text describing the driving scene.

2. @ENTITY_LIST_MODE
   Output Template: Strictly valid JSON using the following schema:
   {
     "detected_entities": [
       [ "Category", "Type", { "KeyAttributes": "Value" }, "Relative_Location" ]
     ]
   }"""
    
    def collate(items: List[Dict]):
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
        
        result = {
            "sample_tokens": tokens,
            "prompt_ids": prompt_batch["input_ids"],
            "prompt_attn": prompt_batch["attention_mask"],
            "answer_ids": ans_batch["input_ids"],
            "answer_attn": ans_batch["attention_mask"],
        }
        
        # Collate camera images if present
        # Check all items for consistency, not just the first one
        if load_images:
            # items[i]["images"] is List[Optional[Tensor]] with 6 views
            # We need to create List[List[Optional[Tensor]]] for batch
            batch_images: List[List[Optional[torch.Tensor]]] = []
            for it in items:
                if "images" in it:
                    batch_images.append(it["images"])
                else:
                    # Fallback: create list of None for missing images
                    batch_images.append([None] * 6)
            result["images"] = batch_images
        
        return result
        
    return collate
