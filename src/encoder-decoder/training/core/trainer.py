"""Main trainer class"""

import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict

from ..data import MixedNuDataset, make_collate, SingleProcessDetSampler
from ..utils import (
    init_dist_if_needed,
    is_main_process,
    Tee,
    set_seed,
    count_trainable_params,
    save_state,
    try_load_state,
    prune_checkpoints_steps,
    plot_loss_curve,
)
from .model_setup import setup_models, create_vat_lidar, setup_optimizer_and_scheduler
from .validation import run_validation, run_inference_sampling
from deepencoder.deepencoder_infer import DEFAULT_VIEW_ORDER, multiview_tokens_from_sample_token


class Trainer:
    """Main training orchestrator"""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.rank, self.local_rank, self.world_size = init_dist_if_needed()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda" and self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
        
        # Logging
        self.tee = None
        self.out_dir = Path(config["out_dir"])
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.tee = Tee(self.out_dir / "train.log")
            sys.stdout = self.tee
        
        print(f"[device] {self.device.type}  fp16={config['fp16']}  GPUs={self.world_size}")
        
        # Set seed
        set_seed(config["seed"])
        
        # Initialize models
        self._setup_models()
        
        # Initialize datasets
        self._setup_datasets()
        
        # Create LiDAR VAT now that we know BEV shape
        self.vat_lidar = create_vat_lidar(self.c_in, self.d_model, config, self.device)
        
        # Print parameter counts
        t_base, a_base, _ = count_trainable_params(self.base)
        t_lidar, a_lidar, _ = count_trainable_params(self.vat_lidar)
        print(f"[param] trainable={t_base + t_lidar:,}")
        
        # Setup DDP
        self._setup_ddp()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Training state
        self.start_epoch = 1
        self.it_resume = 0
        self.global_step = 0
        self.epoch_losses = []
        self.val_losses = []
        self.val_epochs = []
        self.best_val_loss = float("inf")
        self.best_step = None
        
        # Resume if configured
        if config["resume"]:
            self._try_resume()
        
        self.use_amp = config["fp16"] and self.device.type == "cuda"
        
        if is_main_process():
            print(f"[train] epochs={config['epochs']} steps/epoch={self.steps_per_epoch} total_steps={self.total_steps}")
    
    def _setup_models(self):
        """Initialize all models"""
        (
            self.tok,
            self.base,
            _,  # vat_lidar created later
            self.vat_vision,
            self.vision_adapter,
            self.runtime,
            self.nusc,
            self.d_model,
            _,  # c_in
        ) = setup_models(self.config, self.device, is_main_process())
    
    def _setup_datasets(self):
        """Initialize datasets and dataloaders"""
        # Full dataset
        ds_full = MixedNuDataset(
            self.config["jsons"],
            self.config["feature_dirs"],
            target_field=self.config["target_field"],
            max_samples=self.config["max_samples"],
            seed=self.config["seed"],
        )
        
        # Train/val split
        val_size = max(1, int(len(ds_full) * self.config["val_split"]))
        train_size = len(ds_full) - val_size
        
        set_seed(self.config["seed"])
        ds_train, ds_val = torch.utils.data.random_split(
            ds_full, [train_size, val_size], generator=torch.Generator().manual_seed(self.config["seed"])
        )
        
        if is_main_process():
            print(f"[dataset] train={train_size}  val={val_size}")
        
        # Probe BEV shape
        probe = ds_full[0]["bev"]
        self.c_in = int(probe.shape[0])
        
        # Samplers
        sampler_train = (
            DistributedSampler(ds_train, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            if self.world_size > 1
            else SingleProcessDetSampler(ds_train, seed=self.config["seed"], shuffle=True)
        )
        
        # DataLoaders
        self.dl_train = DataLoader(
            ds_train,
            batch_size=self.config["batch_size"],
            shuffle=False,
            sampler=sampler_train,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=make_collate(self.tok, self.config["max_ans_toks"]),
        )
        
        self.dl_val = DataLoader(
            ds_val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=make_collate(self.tok, self.config["max_ans_toks"]),
        )
        
        self.ds_val = ds_val
        self.sampler_train = sampler_train
        self.train_size = train_size
    
    def _setup_ddp(self):
        """Setup distributed data parallel"""
        if self.world_size > 1:
            self.vat_lidar = nn.parallel.DistributedDataParallel(
                self.vat_lidar, device_ids=[self.local_rank], find_unused_parameters=True
            )
            self.base = nn.parallel.DistributedDataParallel(
                self.base, device_ids=[self.local_rank], find_unused_parameters=True
            )
            if self.config["use_vision"]:
                self.vision_adapter = nn.parallel.DistributedDataParallel(
                    self.vision_adapter, device_ids=[self.local_rank], find_unused_parameters=True
                )
                self.vat_vision = nn.parallel.DistributedDataParallel(
                    self.vat_vision, device_ids=[self.local_rank], find_unused_parameters=True
                )
                self.runtime.projector = nn.parallel.DistributedDataParallel(
                    self.runtime.projector, device_ids=[self.local_rank], find_unused_parameters=True
                )
                self.runtime.clip_vit = nn.parallel.DistributedDataParallel(
                    self.runtime.clip_vit, device_ids=[self.local_rank], find_unused_parameters=True
                )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optim, self.sched, self.sched_meta = setup_optimizer_and_scheduler(
            self.base,
            self.vat_lidar,
            self.vat_vision,
            self.vision_adapter,
            self.runtime,
            self.config,
            self.train_size,
            self.world_size,
        )
        
        self.total_steps = self.sched_meta["total_steps"]
        effective_batch_size = self.config["batch_size"] * max(1, self.world_size) * self.config["grad_accum"]
        self.steps_per_epoch = max(1, math.ceil(self.train_size / effective_batch_size))
    
    def _try_resume(self):
        """Try to resume from checkpoint"""
        prev_state, tag = try_load_state(self.out_dir)
        if prev_state is not None:
            if is_main_process():
                print(f"[resume] loading from {tag}")
            
            # Load model states
            vat_lidar_path = self.out_dir / f"vat_lidar_{tag}.pt"
            if vat_lidar_path.exists():
                vat_lidar_model = (
                    self.vat_lidar.module
                    if isinstance(self.vat_lidar, nn.parallel.DistributedDataParallel)
                    else self.vat_lidar
                )
                vat_lidar_model.load_state_dict(torch.load(vat_lidar_path, map_location=self.device))
            
            if self.config["use_vision"]:
                vat_vision_path = self.out_dir / f"vat_vision_{tag}.pt"
                if vat_vision_path.exists():
                    vat_vision_model = (
                        self.vat_vision.module
                        if isinstance(self.vat_vision, nn.parallel.DistributedDataParallel)
                        else self.vat_vision
                    )
                    vat_vision_model.load_state_dict(torch.load(vat_vision_path, map_location=self.device))
                
                va_path = self.out_dir / f"vision_adapter_{tag}.pt"
                proj_path = self.out_dir / f"projector_{tag}.pt"
                
                if va_path.exists():
                    vision_adapter_model = (
                        self.vision_adapter.module
                        if isinstance(self.vision_adapter, nn.parallel.DistributedDataParallel)
                        else self.vision_adapter
                    )
                    vision_adapter_model.load_state_dict(torch.load(va_path, map_location=self.device))
                
                if proj_path.exists():
                    proj_model = (
                        self.runtime.projector.module
                        if isinstance(self.runtime.projector, nn.parallel.DistributedDataParallel)
                        else self.runtime.projector
                    )
                    proj_model.load_state_dict(torch.load(proj_path, map_location=self.device))
            
            # Load optimizer/scheduler
            self.optim.load_state_dict(prev_state["optimizer"])
            self.sched.load_state_dict(prev_state["scheduler"])
            
            # Restore training state
            self.start_epoch = prev_state["epoch"]
            self.it_resume = prev_state["it_in_epoch"]
            self.global_step = prev_state["global_step"]
            self.epoch_losses = prev_state.get("epoch_losses", [])
            self.val_losses = prev_state.get("val_losses", [])
            self.val_epochs = prev_state.get("val_epochs", [])
            self.best_val_loss = prev_state.get("best_loss", float("inf"))
            self.best_step = prev_state.get("best_step", None)
            
            # Restore RNG states
            random.setstate(prev_state["rng"]["py_random"])
            np.random.set_state(prev_state["rng"]["np_random"])
            torch.set_rng_state(prev_state["rng"]["torch"])
            if prev_state["rng"]["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(prev_state["rng"]["torch_cuda"])
            
            if is_main_process():
                print(f"[resume] continuing from epoch {self.start_epoch}, step {self.global_step}")
    
    def _set_epoch(self, epoch: int):
        """Set epoch for samplers"""
        if isinstance(self.sampler_train, (SingleProcessDetSampler, DistributedSampler)):
            self.sampler_train.set_epoch(epoch)
    
    def train(self):
        """Main training loop"""
        # Set models to train mode
        self.base.train()
        self.vat_lidar.train()
        if self.config["use_vision"]:
            self.vision_adapter.train()
            self.vat_vision.train()
            # Use runtime.train() to properly set CLIP/Projector to train mode
            # while keeping SAM frozen in eval mode
            self.runtime.train()
        
        pbar = tqdm(
            total=self.total_steps,
            initial=self.global_step,
            disable=not is_main_process(),
            dynamic_ncols=True,
        )
        
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            self._set_epoch(epoch)
            
            epoch_loss_sum = 0.0
            epoch_count = 0
            
            for it, batch in enumerate(self.dl_train, start=1):
                # Skip if resuming mid-epoch
                if epoch == self.start_epoch and it <= self.it_resume:
                    continue
                
                loss = self._train_step(batch)
                epoch_loss_sum += loss
                epoch_count += 1
                
                if it % self.config["grad_accum"] == 0:
                    self._optimizer_step()
                    self.global_step += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=loss, lr=f"{self.sched.get_last_lr()[0]:.2e}")
                    
                    # Save checkpoint every N steps
                    if (
                        is_main_process()
                        and self.config["save_every_steps"] > 0
                        and self.global_step % self.config["save_every_steps"] == 0
                    ):
                        self._save_checkpoint(f"step{self.global_step}", epoch, it)
                        prune_checkpoints_steps(self.out_dir, self.config["keep_last_n"], self.best_step)
                    
                    if self.global_step >= self.total_steps:
                        break
            
            # Epoch complete
            avg_epoch_loss = epoch_loss_sum / max(1, epoch_count)
            self.epoch_losses.append(avg_epoch_loss)
            
            if is_main_process():
                print(f"\n[epoch {epoch}] train_loss={avg_epoch_loss:.4f}")
            
            # Validation
            if epoch % self.config["validate_every"] == 0:
                self._run_validation(epoch)
            
            # Save latest
            if is_main_process():
                self._save_checkpoint("latest", epoch, len(self.dl_train))
            
            # Run inference sampling
            if is_main_process() and epoch % self.config.get("inference_sampling_every", 3) == 0:
                print(f"\n[inference_sampling] Running at epoch {epoch}")
                run_inference_sampling(
                    self.base,
                    self.vat_lidar,
                    self.vat_vision if self.config["use_vision"] else None,
                    self.vision_adapter if self.config["use_vision"] else None,
                    self.runtime if self.config["use_vision"] else None,
                    self.nusc if self.config["use_vision"] else None,
                    self.tok,
                    self.config,
                    self.out_dir,
                    epoch,
                    self.device,
                    self.ds_train.dataset.token2path,
                    self.best_step,
                )
            
            # Plot
            if is_main_process():
                plot_loss_curve(self.epoch_losses, self.val_losses, self.val_epochs, self.out_dir)
            
            if self.global_step >= self.total_steps:
                break
        
        pbar.close()
        
        if is_main_process():
            print(f"[done] training complete. Best val loss: {self.best_val_loss:.4f} at step {self.best_step}")
            if isinstance(sys.stdout, Tee):
                orig = sys.stdout.stdout
                tee = sys.stdout
                sys.stdout = orig
                tee.close()
    
    def _train_step(self, batch):
        """Single training step"""
        bev = batch["bev"].to(self.device)
        p_ids = batch["prompt_ids"].to(self.device)
        a_ids = batch["answer_ids"].to(self.device)
        sample_tokens = batch["sample_tokens"]
        
        # Vision pipeline
        if self.config["use_vision"]:
            vision_kvs = []
            for tok_str in sample_tokens:
                mv = multiview_tokens_from_sample_token(
                    tok_str, self.nusc, runtime=self.runtime, view_order=DEFAULT_VIEW_ORDER, strict=False
                )
                
                if not mv.get("tokens") or len(mv["tokens"]) != 6:
                    if is_main_process():
                        print(f"[warn] failed to extract 6 view tokens for {tok_str}, using dummy...")
                    dummy_shape = (400, 1280)
                    mv["tokens"] = [torch.zeros(dummy_shape, device=self.device) for _ in range(6)]
                    mv["grid"] = [20, 20]
                
                vt = [t.to(self.device) for t in mv["tokens"]]
                
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    kv = self.vision_adapter(vt)  # [1536, 2048]
                    kv = kv.unsqueeze(0)  # Add batch dimension: [1, 1536, 2048]
                vision_kvs.append(kv)
            
            # Concatenate along batch dimension: [B, 1536, 2048]
            vision_kv = torch.cat(vision_kvs, dim=0)
        else:
            vision_kv = None
        
        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            E = self.base.get_input_embeddings()
            
            def emb(txt):
                ids = self.tok([txt], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                return E(ids)
            
            prefix_lidar = self.vat_lidar(bev) * self.config["prefix_scale"]
            if vision_kv is not None:
                prefix_vision = self.vat_vision(vision_kv) * self.config["prefix_scale"]
            else:
                prefix_vision = None
            
            tok_emb = E(p_ids)
            
            pieces = []
            if prefix_vision is not None:
                pieces += [
                    emb("<vision_start>").expand(prefix_vision.size(0), -1, -1),
                    prefix_vision,
                    emb("<vision_end>").expand(prefix_vision.size(0), -1, -1),
                ]
            
            pieces += [
                emb("<lidar_start>").expand(prefix_lidar.size(0), -1, -1),
                prefix_lidar,
                emb("<lidar_end>").expand(prefix_lidar.size(0), -1, -1),
                tok_emb,
            ]
            
            inp = torch.cat(pieces, dim=1)
            ans_emb = E(a_ids)
            inp = torch.cat([inp, ans_emb], dim=1)
            
            B = inp.size(0)
            total_len = inp.size(1)
            
            labels = torch.full((B, total_len), -100, dtype=torch.long, device=self.device)
            labels[:, -a_ids.size(1) :] = a_ids
            
            attn = torch.ones((B, total_len), dtype=torch.long, device=self.device)
            
            out = self.base(inputs_embeds=inp, attention_mask=attn, labels=labels)
            loss = out.loss / self.config["grad_accum"]
        
        loss.backward()
        return loss.item() * self.config["grad_accum"]
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping"""
        torch.nn.utils.clip_grad_norm_(self.vat_lidar.parameters(), self.config["clip_norm"])
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.base.parameters() if p.requires_grad], self.config["clip_norm"]
        )
        if self.config["use_vision"]:
            torch.nn.utils.clip_grad_norm_(self.vision_adapter.parameters(), self.config["clip_norm"])
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.runtime.projector.parameters() if p.requires_grad], self.config["clip_norm"]
            )
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.runtime.clip_vit.parameters() if p.requires_grad], self.config["clip_norm"]
            )
            torch.nn.utils.clip_grad_norm_(self.vat_vision.parameters(), self.config["clip_norm"])
        
        self.optim.step()
        self.sched.step()
        self.optim.zero_grad(set_to_none=True)
    
    def _run_validation(self, epoch: int):
        """Run validation"""
        if is_main_process():
            print(f"[validation] epoch {epoch}...")
        
        val_loss = run_validation(
            self.dl_val,
            self.device,
            self.tok,
            self.base,
            self.vat_lidar,
            self.vat_vision if self.config["use_vision"] else None,
            self.vision_adapter if self.config["use_vision"] else None,
            self.runtime if self.config["use_vision"] else None,
            self.nusc if self.config["use_vision"] else None,
            self.config,
        )
        
        if is_main_process():
            print(f"[validation] epoch={epoch} val_loss={val_loss:.4f}")
            self.val_losses.append(val_loss)
            self.val_epochs.append(epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_step = self.global_step
                print(f"[best-val] new best: {self.best_val_loss:.4f} at step {self.best_step}")
                self._save_best_checkpoint()
    
    def _save_checkpoint(self, tag: str, epoch: int, it_in_epoch: int):
        """Save checkpoint"""
        save_state(
            self.out_dir,
            tag,
            step=self.global_step,
            epoch=epoch,
            it_in_epoch=it_in_epoch,
            global_step=self.global_step,
            epoch_losses=self.epoch_losses,
            best_loss=self.best_val_loss,
            best_step=self.best_step,
            optim=self.optim,
            sched=self.sched,
            vat_lidar=self.vat_lidar,
            vat_vision=self.vat_vision if self.config["use_vision"] else None,
            base=self.base,
            clip_vit=self.runtime.clip_vit if self.config["use_vision"] else None,
            vision_adapter=self.vision_adapter if self.config["use_vision"] else None,
            projector=self.runtime.projector if self.config["use_vision"] else None,
            sched_meta=self.sched_meta,
            config=self.config,
            val_losses=self.val_losses,
            val_epochs=self.val_epochs,
        )
    
    def _save_best_checkpoint(self):
        """Save best model checkpoint"""
        def unwrap(model):
            return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        
        torch.save(unwrap(self.vat_lidar).state_dict(), self.out_dir / "vat_lidar_best.pt")
        
        if self.config["use_vision"]:
            torch.save(unwrap(self.vat_vision).state_dict(), self.out_dir / "vat_vision_best.pt")
        
        unwrap(self.base).save_pretrained(self.out_dir / "qwen2_lora_adapter_best")
        
        if self.config["use_vision"]:
            torch.save(unwrap(self.vision_adapter).state_dict(), self.out_dir / "vision_adapter_best.pt")
            torch.save(unwrap(self.runtime.projector).state_dict(), self.out_dir / "projector_best.pt")
            unwrap(self.runtime.clip_vit).save_pretrained(self.out_dir / "clip_lora_adapter_best")
            print(f"[best-val] saved all vision components")
