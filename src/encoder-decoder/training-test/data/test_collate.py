"""Tests for collate function"""

import pytest
import torch
from unittest.mock import Mock
from training.data.collate import make_collate


class TestMakeCollate:
    """Tests for make_collate function"""
    
    def setup_method(self):
        """Setup mock tokenizer for tests"""
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template = Mock(
            side_effect=lambda msgs, **kwargs: f"System: {msgs[0]['content']}\nUser: {msgs[1]['content']}"
        )
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
    
    def test_collate_basic_functionality(self):
        """Test basic collate functionality"""
        # Setup tokenizer to return proper tensors
        def mock_tokenizer_call(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.side_effect = mock_tokenizer_call
        
        collate_fn = make_collate(self.tokenizer, max_ans_toks=50)
        
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": "sample_token_1",
                "question": "What do you see?",
                "answer": "I see a car"
            },
            {
                "bev": torch.randn(128, 64, 64),
                "token": "sample_token_2",
                "question": "Where is the truck?",
                "answer": "The truck is ahead"
            }
        ]
        
        batch = collate_fn(items)
        
        assert "bev" in batch
        assert "sample_tokens" in batch
        assert "prompt_ids" in batch
        assert "prompt_attn" in batch
        assert "answer_ids" in batch
        assert "answer_attn" in batch
        
        assert batch["bev"].shape[0] == 2  # Batch size
        assert len(batch["sample_tokens"]) == 2
    
    def test_collate_stacks_bevs(self):
        """Test that BEV tensors are properly stacked"""
        def mock_tokenizer_call(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.side_effect = mock_tokenizer_call
        
        collate_fn = make_collate(self.tokenizer, max_ans_toks=50)
        
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token1",
                "question": "Q1",
                "answer": "A1"
            },
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token2",
                "question": "Q2",
                "answer": "A2"
            },
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token3",
                "question": "Q3",
                "answer": "A3"
            }
        ]
        
        batch = collate_fn(items)
        
        assert batch["bev"].shape == (3, 128, 64, 64)
        assert isinstance(batch["bev"], torch.Tensor)
    
    def test_collate_preserves_sample_tokens(self):
        """Test that sample tokens are preserved in correct order"""
        def mock_tokenizer_call(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.side_effect = mock_tokenizer_call
        
        collate_fn = make_collate(self.tokenizer, max_ans_toks=50)
        
        tokens = ["token_a", "token_b", "token_c"]
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": tok,
                "question": f"Q{i}",
                "answer": f"A{i}"
            }
            for i, tok in enumerate(tokens)
        ]
        
        batch = collate_fn(items)
        
        assert batch["sample_tokens"] == tokens
    
    def test_collate_applies_chat_template(self):
        """Test that chat template is applied to questions"""
        call_count = [0]
        
        def mock_apply_template(msgs, **kwargs):
            call_count[0] += 1
            return f"Formatted: {msgs[1]['content']}"
        
        def mock_tokenizer_call(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.apply_chat_template = mock_apply_template
        self.tokenizer.side_effect = mock_tokenizer_call
        
        collate_fn = make_collate(self.tokenizer, max_ans_toks=50, system_prompt="You are an AI")
        
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token1",
                "question": "What is this?",
                "answer": "It's a car"
            }
        ]
        
        batch = collate_fn(items)
        
        # Chat template should be called once per item
        assert call_count[0] == 1
    
    def test_collate_uses_custom_system_prompt(self):
        """Test that custom system prompt is used"""
        prompts_used = []
        
        def mock_apply_template(msgs, **kwargs):
            prompts_used.append(msgs[0]['content'])
            return f"Formatted: {msgs[1]['content']}"
        
        def mock_tokenizer_call(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.apply_chat_template = mock_apply_template
        self.tokenizer.side_effect = mock_tokenizer_call
        
        custom_prompt = "You are a specialized driving assistant"
        collate_fn = make_collate(self.tokenizer, max_ans_toks=50, system_prompt=custom_prompt)
        
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token1",
                "question": "Q",
                "answer": "A"
            }
        ]
        
        batch = collate_fn(items)
        
        assert prompts_used[0] == custom_prompt
    
    def test_collate_uses_default_system_prompt(self):
        """Test that default system prompt is used when none provided"""
        prompts_used = []
        
        def mock_apply_template(msgs, **kwargs):
            prompts_used.append(msgs[0]['content'])
            return f"Formatted"
        
        def mock_tokenizer_call(texts, **kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.apply_chat_template = mock_apply_template
        self.tokenizer.side_effect = mock_tokenizer_call
        
        collate_fn = make_collate(self.tokenizer, max_ans_toks=50)
        
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token1",
                "question": "Q",
                "answer": "A"
            }
        ]
        
        batch = collate_fn(items)
        
        # Should use default prompt containing "autonomous driving"
        assert "autonomous driving" in prompts_used[0].lower()
    
    def test_collate_max_ans_toks_parameter(self):
        """Test that max_ans_toks is passed to tokenizer"""
        max_lens_used = []
        
        def mock_tokenizer_call(texts, **kwargs):
            if "max_length" in kwargs:
                max_lens_used.append(kwargs["max_length"])
            batch_size = len(texts)
            return {
                "input_ids": torch.randint(0, 1000, (batch_size, 10)),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long)
            }
        
        self.tokenizer.apply_chat_template = lambda msgs, **kw: "formatted"
        self.tokenizer.side_effect = mock_tokenizer_call
        
        max_ans_toks = 128
        collate_fn = make_collate(self.tokenizer, max_ans_toks=max_ans_toks)
        
        items = [
            {
                "bev": torch.randn(128, 64, 64),
                "token": "token1",
                "question": "Q",
                "answer": "A"
            }
        ]
        
        batch = collate_fn(items)
        
        # Should be called twice: once for prompts, once for answers
        # Answer tokenization should use max_ans_toks
        assert max_ans_toks in max_lens_used


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
