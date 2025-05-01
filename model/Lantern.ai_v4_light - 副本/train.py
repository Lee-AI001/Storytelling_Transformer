import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import math
from lion_pytorch import Lion
from model.transformer import StoryTellerTransformer
from dataloader.dataset import StoryDataset, collate_fn, print_training_data_example
from dataloader.tokenizer_utils import load_tokenizer, get_label2id, train_tokenizer
from model.generate import generate_multiple_stories
import config
from torchmetrics.text.rouge import ROUGEScore
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from rouge_score import rouge_scorer
import random 
import torch.nn.functional as F

# ========================== DYNAMIC BATCH SCHEDULER ==========================

class DynamicBatchScheduler:
    def __init__(self, target_vram_usage=0.9, min_batch_size=8, max_batch_size=128):
        self.target_vram_usage = target_vram_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

    def estimate_batch_size(self, model, dataset, max_len, device, mixed_precision="none"):
        if not torch.cuda.is_available():
            logging.info("Dynamic batch sizing disabled on CPU, using default batch size")
            return config.BATCH_SIZE

        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        try:
            sample = dataset[0]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            target_ids = sample["target_ids"].unsqueeze(0).to(device)
            attention_mask = torch.tril(torch.ones(1, config.NHEAD, max_len, max_len, device=device))
            padding_mask = (input_ids == config.PAD_IDX).to(device)

            model.eval()
            dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=mixed_precision != "none"):
                with torch.no_grad():
                    outputs = model(input_ids, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask)
            memory_per_sample = (torch.cuda.memory_allocated(device) / 1024**3 - initial_memory) / input_ids.size(0)
            model.train()

            total_vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
            available_vram = total_vram * self.target_vram_usage
            memory_per_sample *= 0.6 if mixed_precision != "none" else 1.0
            estimated_batch_size = int(available_vram / max(1e-6, memory_per_sample))
            batch_size = max(self.min_batch_size, min(self.max_batch_size, estimated_batch_size))
            logging.info(f"Estimated batch size: {batch_size} (memory/sample: {memory_per_sample:.2f}GB, available VRAM: {available_vram:.2f}GB, precision: {mixed_precision})")
            return batch_size
        except Exception as e:
            logging.warning(f"Failed to estimate batch size: {e}, using min batch size")
            return self.min_batch_size

# ========================== SGDR SCHEDULER ==========================

class SGDRScheduler:
    def __init__(self, optimizer, min_lr, max_lr, cycle_length, warmup_epochs, initial_cycle_mult=1):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.cycle_mult = initial_cycle_mult
        self.current_cycle = 0
        self.cycle_progress = 0
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

    def step(self):
        self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            new_lr = [self.max_lr * progress for _ in self.optimizer.param_groups]
        else:
            self.cycle_progress += 1
            if self.cycle_progress >= self.cycle_length:
                self.current_cycle += 1
                self.cycle_progress = 0
                self.cycle_length = int(self.cycle_length * self.cycle_mult)
            progress = self.cycle_progress / self.cycle_length
            new_lr = [
                self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
                for _ in self.optimizer.param_groups
            ]

        for param_group, lr in zip(self.optimizer.param_groups, new_lr):
            param_group['lr'] = lr
        return new_lr

    def get_state_dict(self):
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'cycle_length': self.cycle_length,
            'warmup_epochs': self.warmup_epochs,
            'current_epoch': self.current_epoch,
            'cycle_mult': self.cycle_mult,
            'current_cycle': self.current_cycle,
            'cycle_progress': self.cycle_progress,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        self.min_lr = state_dict['min_lr']
        self.max_lr = state_dict['max_lr']
        self.cycle_length = state_dict['cycle_length']
        self.warmup_epochs = state_dict.get('warmup_epochs', 0)
        self.current_epoch = state_dict['current_epoch']
        self.cycle_mult = state_dict['cycle_mult']
        self.current_cycle = state_dict['current_cycle']
        self.cycle_progress = state_dict['cycle_progress']
        self.base_lrs = state_dict['base_lrs']

# ========================== PLOT METRICS ==========================

def plot_metrics(epochs, train_losses, val_losses, perplexities, rouges, genre_losses, learning_rates, weight_decays, output_path):
    if not epochs:
        logging.warning("No metrics data to plot")
        print("⚠️ No metrics data to plot")
        return
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, perplexities, label="Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title("Perplexity")
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, rouges, label="ROUGE-L")
    plt.xlabel("Epoch")
    plt.ylabel("ROUGE-L")
    plt.legend()
    plt.title("ROUGE-L")
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, genre_losses, label="Genre Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Genre Loss")
    plt.legend()
    plt.title("Genre Loss")
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, learning_rates, label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.title("Learning Rate")
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, weight_decays, label="Weight Decay")
    plt.xlabel("Epoch")
    plt.ylabel("Weight Decay")
    plt.legend()
    plt.title("Weight Decay")
    
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved metrics plot to {output_path}")
        print(f"✅ Metrics plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics plot: {e}")
        print(f"❌ Metrics plot save error: {e}")

# ========================== EVALUATION ==========================

def evaluate(model, data_loader, criterion, genre_criterion, device, vocab_size, tokenizer, pad_idx=0):
    if len(data_loader.dataset) == 0:
        print("⚠️ Validation dataset is empty")
        return {"loss": float("inf"), "perplexity": float("inf"), "rouge": 0.0, "genre_loss": 0.0}

    model.eval()
    total_loss = 0
    total_genre_loss = 0
    total_batches = 0
    hypotheses = []
    references = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    dev_samples = list(data_loader.dataset)
    random.shuffle(dev_samples)
    selected_samples = dev_samples[:min(10, len(dev_samples))]
    print(f"🧹 Starting evaluation with {len(data_loader)} batches, ROUGE on {len(selected_samples)} samples...")

    with torch.no_grad():
        for sample in selected_samples:
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            ref_tokens = [t for t in sample['input_ids'].tolist() if t not in [pad_idx, config.EOS_IDX]]
            reference = tokenizer.decode(ref_tokens).strip()
            if not reference:
                logging.warning("Skipping empty reference")
                continue

            generated_ids = input_ids.clone()
            for _ in range(config.MAX_GEN_LEN):
                outputs = model(generated_ids, apply_dropout=False)
                logits = outputs[0] if config.USE_GENRE_PREDICTION else outputs
                logits = logits[:, -1, :] / config.TEMPERATURE
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config.TOP_P
                sorted_indices_to_remove[..., :config.TOP_K] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                if next_token.item() == config.EOS_IDX:
                    break
            hyp_tokens = [t for t in generated_ids[0].cpu().tolist() if t not in [pad_idx, config.EOS_IDX]]
            hypothesis = tokenizer.decode(hyp_tokens).strip()
            if not hypothesis:
                logging.warning("Skipping empty hypothesis")
                continue
            hypotheses.append(hypothesis)
            references.append(reference)

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask, apply_dropout=False)
            logits = outputs if not config.USE_GENRE_PREDICTION else outputs[0]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("⚠️ Found NaN/Inf in logits, skipping batch")
                continue

            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            mask = target_ids != pad_idx
            loss = (loss * mask.view(-1)).sum() / mask.sum()

            total_loss += loss.item()
            if config.USE_GENRE_PREDICTION:
                genre_loss = genre_criterion(outputs[1], batch["genre_id"].to(device))
                total_genre_loss += genre_loss.item()

            total_batches += 1

    if total_batches == 0:
        print("⚠️ No valid batches during evaluation")
        return {"loss": float("inf"), "perplexity": float("inf"), "rouge": 0.0, "genre_loss": 0.0}

    avg_loss = total_loss / total_batches
    avg_genre_loss = total_genre_loss / total_batches if config.USE_GENRE_PREDICTION else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    if not hypotheses or not references:
        print("⚠️ No valid hypotheses collected for ROUGE evaluation")
        logging.warning(f"No valid hypotheses/references: {len(hypotheses)} hypotheses, {len(references)} references")
        return {"loss": avg_loss, "perplexity": perplexity, "rouge": 0.0, "genre_loss": avg_genre_loss}

    print(f"📏 Computing ROUGE for {len(hypotheses)} hypothesis/reference pairs...")
    rouge_scores = []
    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        try:
            score = scorer.score(ref, hyp)['rougeL'].fmeasure
            rouge_scores.append(score)
            if i < 3:
                logging.info(f"Sample {i+1} - ROUGE-L: {score:.4f}, Hyp: {hyp[:50]}..., Ref: {ref[:50]}...")
        except Exception as e:
            logging.warning(f"ROUGE error at sample {i}: {e}")
            continue

    rouge_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    print(f"✅ Average ROUGE-L: {rouge_score:.4f} from {len(rouge_scores)} valid samples")

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "rouge": rouge_score,
        "genre_loss": avg_genre_loss
    }

# ========================== CHECKPOINTING ==========================

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, scheduler=None, warmup_scheduler=None, scaler=None, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "transformer_best.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}.pth")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "genre_frozen": genre_frozen,
    }
    if scheduler:
        if hasattr(scheduler, 'get_state_dict'):
            checkpoint["scheduler_state_dict"] = scheduler.get_state_dict()
        elif hasattr(scheduler, 'state_dict'):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        else:
            logging.warning("Scheduler has no state_dict or get_state_dict method, skipping.")
    if warmup_scheduler and hasattr(warmup_scheduler, 'state_dict'):
        checkpoint["warmup_scheduler_state_dict"] = warmup_scheduler.state_dict()
    
    try:
        torch.save(checkpoint, checkpoint_path)
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
            logging.info(f"Saved checkpoint: {checkpoint_path} (Size: {os.path.getsize(checkpoint_path) / 1024**2:.2f} MB)")
            print(f"✅ Checkpoint saved: {checkpoint_path}")
        else:
            logging.error(f"Checkpoint file {checkpoint_path} is empty or missing after save")
            print(f"❌ Checkpoint save error: File is empty or missing")
    except Exception as e:
        logging.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
        print(f"❌ Checkpoint save error: {e}")

def load_checkpoint(model, optimizer, scheduler, warmup_scheduler, scaler, checkpoint_dir, device):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("transformer_epoch_") and f.endswith(".pth")]
    if not checkpoint_files:
        logging.warning(f"No checkpoints found in {checkpoint_dir}, starting from scratch")
        print(f"⚠️ No checkpoints found")
        return 0, float("inf")

    try:
        latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict") is not None:
            if isinstance(scheduler, SGDRScheduler):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            elif hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                logging.warning("Scheduler has no load_state_dict method, skipping.")
        if warmup_scheduler and checkpoint.get("warmup_scheduler_state_dict") is not None:
            warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        val_loss = checkpoint["val_loss"]
        global genre_frozen
        genre_frozen = checkpoint.get("genre_frozen", False)
        if genre_frozen and config.USE_GENRE_PREDICTION:
            for param in model.genre_head.parameters():
                param.requires_grad = False
            logging.info("Restored genre_frozen status: True")
        logging.info(f"Resumed from checkpoint: {checkpoint_path} (epoch {epoch})")
        print(f"✅ Resumed from epoch {epoch} with val loss {val_loss:.4f}")
        return epoch, val_loss
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        print(f"❌ Checkpoint load error: {e}")
        return 0, float("inf")

# ========================== LOGGING ==========================

def save_generated_story(story, prompt, index, output_file_path):
    try:
        with open(output_file_path, "a", encoding="utf-8") as f:
            f.write(f"Prompt {index}:\n{prompt}\n")
            f.write(f"Story:\n{story}\n\n")
        logging.info(f"Saved story {index}")
        print(f"✅ Story {index} saved")
    except IOError as e:
        logging.error(f"Failed to save story {index}: {e}")
        print(f"❌ Story {index} save error: {e}")

def log_training_metrics(epoch, train_loss, dev_loss, perplexity, rouge, genre_loss, file_path, optimizer_name, lr, weight_decay):
    try:
        file_exists = os.path.exists(file_path)
        is_empty = not file_exists or os.path.getsize(file_path) == 0
        with open(file_path, "a", encoding="utf-8") as f:
            if is_empty:
                f.write(f"Training Metrics\nOptimizer: {optimizer_name}\n")
                f.write("Epoch  |  Train Loss  |  Dev Loss  |  Perplexity  |  ROUGE-L  |  Genre Loss  |  Learning Rate  |  Weight Decay\n")
                f.write("-" * 90 + "\n")
            f.write(f"{epoch:5d} |    {train_loss:.4f}    |    {dev_loss:.4f}    |    {perplexity:.2f}    |    {rouge:.4f}    |    {genre_loss:.4f}    |    {lr:.6f}    |    {weight_decay:.6f}\n\n")
        logging.info(f"Metrics logged for epoch {epoch}")
        print(f"✅ Metrics logged for epoch {epoch}")
    except IOError as e:
        logging.error(f"Failed to log metrics for epoch {epoch}: {e}")
        print(f"❌ Metrics log error: {e}")

# ========================== SCHEDULER ==========================

def create_scheduler(optimizer, epochs):
    scheduler = None
    warmup_scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - config.WARMUP_EPOCHS,
                eta_min=config.SGDR_MIN_LR
            )
            if config.WARMUP_EPOCHS > 0:
                warmup_lambda = lambda epoch: min((epoch + 1) / config.WARMUP_EPOCHS, 1.0)
                warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
                logging.info(f"Initialized Cosine scheduler with warmup={config.WARMUP_EPOCHS} epochs")
                print(f"🔄 Cosine scheduler with warmup initialized")
        elif config.SCHEDULER_TYPE == "sgdr":
            scheduler = SGDRScheduler(
                optimizer,
                min_lr=config.SGDR_MIN_LR,
                max_lr=config.SGDR_MAX_LR,
                cycle_length=config.SGDR_CYCLE_LENGTH
            )
            logging.info(f"Initialized SGDR scheduler with cycle_length={config.SGDR_CYCLE_LENGTH}, warmup={config.WARMUP_EPOCHS} epochs")
            print(f"🔄 SGDR scheduler with cycle_length={config.SGDR_CYCLE_LENGTH} initialized")
            if config.WARMUP_EPOCHS > 0:
                warmup_lambda = lambda epoch: min((epoch + 1) / config.WARMUP_EPOCHS, 1.0)
                warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
        else:
            logging.warning(f"Unknown SCHEDULER_TYPE: {config.SCHEDULER_TYPE}, no scheduler used")
            print(f"⚠️ Unknown SCHEDULER_TYPE: {config.SCHEDULER_TYPE}")
    return scheduler, warmup_scheduler

def get_context_window_params(epoch):
    """Get context window parameters for the current epoch."""
    schedule = config.DYNAMIC_CONTEXT_SCHEDULE
    if not schedule:
        logging.warning("DYNAMIC_CONTEXT_SCHEDULE is empty, using default parameters")
        return config.MAX_LEN, config.SLIDING_STEP, config.MAX_CHUNKS
    for start_epoch, max_len, sliding_step, max_chunks in schedule:
        if epoch >= start_epoch:
            return max_len, sliding_step, max_chunks
    return schedule[0][1], schedule[0][2], schedule[0][3]

# ========================== MODEL INTRO ==========================

def save_model_intro(storage_dir, model, config):
    """Save a model introduction file with comprehensive model and training details."""
    intro_path = os.path.join(storage_dir, "model_intro.txt")
    model_info = {
        "Model Name": "StoryTellerTransformer",
        "Vocabulary Size": config.VOCAB_SIZE,
        "Embedding Dimension (d_model)": config.D_MODEL,
        "Number of Attention Heads (nhead)": config.NHEAD,
        "Number of Decoder Layers": config.NUM_LAYERS,
        "Feedforward Dimension": config.DIM_FEEDFORWARD,
        "Dropout Rate": config.DROPOUT,
        "Embedding Dropout Rate": config.EMBED_DROPOUT,
        "Genre Prediction Enabled": config.USE_GENRE_PREDICTION,
        "Number of Genres": config.NUM_GENRES if config.USE_GENRE_PREDICTION else "N/A",
        "Optimizer": config.USE_OPT,
        "Learning Rate": config.LEARNING_RATE,
        "Weight Decay": config.WEIGHT_DECAY,
        "Scheduler Type": config.SCHEDULER_TYPE if config.USE_SCHEDULER else "None",
        "SGDR Cycle Length": config.SGDR_CYCLE_LENGTH if config.SCHEDULER_TYPE == "sgdr" else "N/A",
        "SGDR Min LR": config.SGDR_MIN_LR if config.SCHEDULER_TYPE == "sgdr" else "N/A",
        "SGDR Max LR": config.SGDR_MAX_LR if config.SCHEDULER_TYPE == "sgdr" else "N/A",
        "Warmup Epochs": config.WARMUP_EPOCHS,
        "Mixed Precision": config.MIXED_PRECISION,
        "Max Epochs": config.EPOCHS,
        "Batch Size": config.BATCH_SIZE,
        "Dynamic Batch": config.DYNAMIC_BATCH,
        "Target VRAM Usage": config.TARGET_VRAM_USAGE if config.DYNAMIC_BATCH else "N/A",
        "Min Batch Size": config.MIN_BATCH_SIZE if config.DYNAMIC_BATCH else "N/A",
        "Max Batch Size": config.MAX_BATCH_SIZE if config.DYNAMIC_BATCH else "N/A",
        "Gradient Accumulation Steps": config.GRAD_ACCUM_STEPS,
        "Max Gradient Norm": config.MAX_GRAD_NORM,
        "Label Smoothing": config.LABEL_SMOOTHING,
        "Genre Loss Weight": config.GENRE_LOSS_WEIGHT,
        "Dynamic Context Schedule": config.DYNAMIC_CONTEXT_SCHEDULE,
        "Tokenizer Path": config.TOKENIZER_PATH,
        "Checkpoint Directory": config.CHECKPOINT_DIR,
        "Generated Stories Path": config.GENERATED_PATH,
        "Training Log Path": config.LOG_PATH,
        "Data File": config.DATA_FILE,
        "Model Architecture Path": config.MODEL_ARCH_PATH,
        "Training Texts Path": config.TRAIN_TXT_PATH,
        "Genres": config.GENRES,
    }
    
    try:
        with open(intro_path, "w", encoding="utf-8") as f:
            f.write("StoryTellerTransformer Model Introduction\n")
            f.write("=" * 50 + "\n\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        if os.path.exists(intro_path) and os.path.getsize(intro_path) > 0:
            logging.info(f"Saved model introduction: {intro_path}")
            print(f"✅ Model introduction saved: {intro_path}")
        else:
            logging.error(f"Model introduction file {intro_path} is empty or missing after save")
            print(f"❌ Model introduction save error: File is empty or missing")
    except Exception as e:
        logging.error(f"Failed to save model introduction: {e}")
        print(f"❌ Model introduction save error: {e}")

# ========================== TRAINING LOOP ==========================

def train():
    global genre_frozen
    logging.info("Initializing training...")
    print("🚀 Starting training...")

    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        logging.info(f"GPU: {gpu_name}, CUDA Version: {cuda_version}")
    else:
        logging.warning("No GPU detected, training will be slow")

    config.save_model_architecture()
    save_model_intro(config.STORAGE_DIR, None, config)

    os.makedirs(os.path.dirname(config.TOKENIZER_PATH), exist_ok=True)
    
    if not os.path.exists(config.TOKENIZER_PATH) or os.path.getsize(config.TOKENIZER_PATH) < 1024:
        logging.info(f"Tokenizer missing or invalid at {config.TOKENIZER_PATH}, training new tokenizer...")
        if not os.path.exists(config.TRAIN_TXT_PATH) or os.path.getsize(config.TRAIN_TXT_PATH) == 0:
            logging.info(f"Training texts not found or empty at {config.TRAIN_TXT_PATH}, creating...")
            config.write_train_texts(config.train_data, config.TRAIN_TXT_PATH)
            if not os.path.exists(config.TRAIN_TXT_PATH) or os.path.getsize(config.TRAIN_TXT_PATH) == 0:
                logging.error(f"Failed to create valid training texts at {config.TRAIN_TXT_PATH}")
                raise ValueError(f"Training texts file {config.TRAIN_TXT_PATH} is empty or could not be created")
        try:
            train_tokenizer(config.TRAIN_TXT_PATH, config.TOKENIZER_PREFIX, config.VOCAB_SIZE, config.story_labels)
        except Exception as e:
            logging.error(f"Failed to train tokenizer: {e}")
            raise RuntimeError(f"Tokenizer training failed: {e}")
    
    try:
        tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise RuntimeError(f"Cannot load tokenizer from {config.TOKENIZER_PATH}: {e}")

    label2id = get_label2id(tokenizer, config.story_labels)

    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != config.VOCAB_SIZE:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) != config.VOCAB_SIZE ({config.VOCAB_SIZE})")
        config.VOCAB_SIZE = tokenizer_vocab_size
        vocab_size = tokenizer_vocab_size
    else:
        vocab_size = config.VOCAB_SIZE

    model = StoryTellerTransformer(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        embed_dropout=config.EMBED_DROPOUT
    ).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX, reduction="none", label_smoothing=config.LABEL_SMOOTHING)
    genre_criterion = nn.CrossEntropyLoss() if config.USE_GENRE_PREDICTION else None

    if config.USE_OPT.lower() == "lion":
        optimizer = Lion(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        optimizer_name = "Lion"
    elif config.USE_OPT.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        optimizer_name = "AdamW"
    else:
        raise ValueError(f"Invalid USE_OPT: {config.USE_OPT}. Must be 'lion' or 'adamw'.")

    scheduler = SGDRScheduler(
        optimizer,
        min_lr=config.SGDR_MIN_LR,
        max_lr=config.SGDR_MAX_LR,
        cycle_length=5,
        warmup_epochs=config.WARMUP_EPOCHS
    ) if config.USE_SCHEDULER and config.SCHEDULER_TYPE == "sgdr" else None
    warmup_scheduler = None
    if config.SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.SGDR_MIN_LR
        )
        if config.WARMUP_EPOCHS > 0:
            warmup_lambda = lambda epoch: min((epoch + 1) / config.WARMUP_EPOCHS, 1.0)
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    scaler = torch.amp.GradScaler('cuda') if config.DEVICE.type == "cuda" and config.MIXED_PRECISION != "none" else None

    batch_scheduler = DynamicBatchScheduler(
        target_vram_usage=config.TARGET_VRAM_USAGE,
        min_batch_size=config.MIN_BATCH_SIZE,
        max_batch_size=config.MAX_BATCH_SIZE
    ) if config.DYNAMIC_BATCH else None

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, warmup_scheduler, scaler, config.CHECKPOINT_DIR, config.DEVICE)
    if start_epoch > 0:
        start_epoch += 1

    epochs_no_improve = 0
    best_genre_loss = float("inf")
    genre_epochs_no_improve = 0
    genre_frozen = False
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "transformer_best.pth")

    epochs = []
    train_losses = []
    val_losses = []
    perplexities = []
    rouges = []
    genre_losses = []
    learning_rates = []
    weight_decays = []

    current_max_len = current_sliding_step = current_max_chunks = None
    train_dataset = dev_dataset = train_loader = dev_loader = None

    for epoch in range(start_epoch, config.EPOCHS):
        try:
            max_len, sliding_step, max_chunks = get_context_window_params(epoch)
            if (train_dataset is None or 
                max_len != current_max_len or 
                sliding_step != current_sliding_step or 
                max_chunks != current_max_chunks):
                logging.info(f"Epoch {epoch+1}: Updating context window - MAX_LEN={max_len}, SLIDING_STEP={sliding_step}, MAX_CHUNKS={max_chunks}")
                
                train_dataset = StoryDataset(
                    data=config.train_data,
                    tokenizer=tokenizer,
                    label2id=label2id,
                    max_tokens=max_len,
                    step=sliding_step,
                    max_chunks=max_chunks
                )
                dev_dataset = StoryDataset(
                    data=config.dev_data,
                    tokenizer=tokenizer,
                    label2id=label2id,
                    max_tokens=max_len,
                    step=sliding_step,
                    max_chunks=max_chunks
                )

                if len(train_dataset) == 0:
                    logging.error("Training dataset is empty")
                    raise ValueError("Training dataset is empty")
                if len(dev_dataset) == 0:
                    logging.warning("Validation dataset is empty, skipping validation")

                batch_size = batch_scheduler.estimate_batch_size(model, train_dataset, max_len, config.DEVICE, config.MIXED_PRECISION) if batch_scheduler else config.BATCH_SIZE
                logging.info(f"Epoch {epoch+1}: Using batch size {batch_size} (effective batch size: {batch_size * config.GRAD_ACCUM_STEPS})")

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=lambda b: collate_fn(b, pad_idx=config.PAD_IDX, nhead=config.NHEAD)
                )
                dev_loader = DataLoader(
                    dev_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda b: collate_fn(b, pad_idx=config.PAD_IDX, nhead=config.NHEAD)
                )

                if len(train_loader) == 0:
                    logging.error("Training data loader is empty")
                    raise ValueError("Training data loader is empty")

                current_max_len = max_len
                current_sliding_step = sliding_step
                current_max_chunks = max_chunks

                test_loader = DataLoader(
                    train_dataset,
                    batch_size=2,
                    shuffle=True,
                    collate_fn=lambda b: collate_fn(b, pad_idx=config.PAD_IDX, nhead=config.NHEAD)
                )
                print_training_data_example(test_loader, tokenizer, pad_idx=config.PAD_IDX)

            model.train()
            total_train_loss = 0
            total_genre_loss = 0
            train_steps = 0
            accumulated_steps = 0

            dtype = torch.float16 if config.MIXED_PRECISION == "fp16" else torch.bfloat16 if config.MIXED_PRECISION == "bf16" else torch.float32

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(config.DEVICE)
                target_ids = batch["target_ids"].to(config.DEVICE)
                padding_mask = batch["padding_mask"].to(config.DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=config.MIXED_PRECISION != "none"):
                    outputs = model(
                        input_ids,
                        tgt_mask=None,
                        tgt_key_padding_mask=padding_mask,
                        apply_dropout=True
                    )

                    if config.USE_GENRE_PREDICTION and config.USE_R_DROP:
                        outputs2 = model(
                            input_ids,
                            tgt_mask=None,
                            tgt_key_padding_mask=padding_mask,
                            apply_dropout=True
                        )
                        logits1, genre_logits1 = outputs
                        logits2, genre_logits2 = outputs2
                        vocab_size = logits1.shape[-1]

                        if torch.isnan(logits1).any() or torch.isinf(logits1).any() or \
                           torch.isnan(logits2).any() or torch.isinf(logits2).any():
                            logging.warning("NaN or inf logits detected in training batch, skipping")
                            continue

                        loss1 = criterion(logits1.view(-1, vocab_size), target_ids.view(-1))
                        loss2 = criterion(logits2.view(-1, vocab_size), target_ids.view(-1))
                        mask = target_ids != config.PAD_IDX
                        if mask.sum() > 0:
                            loss1 = (loss1 * mask.view(-1)).sum() / mask.sum()
                            loss2 = (loss2 * mask.view(-1)).sum() / mask.sum()
                        else:
                            logging.warning("All tokens are padding, skipping batch")
                            continue
                        lm_loss = 0.5 * (loss1 + loss2)

                        genre_loss = genre_criterion(genre_logits1, batch["genre_id"].to(config.DEVICE))
                        genre_loss += genre_criterion(genre_logits2, batch["genre_id"].to(config.DEVICE))
                        genre_loss *= 0.5

                        log_probs1 = torch.log_softmax(logits1, dim=-1)
                        log_probs2 = torch.log_softmax(logits2, dim=-1)
                        kl_loss = nn.functional.kl_div(
                            log_probs1,
                            log_probs2,
                            reduction='batchmean',
                            log_target=True
                        ) + nn.functional.kl_div(
                            log_probs2,
                            log_probs1,
                            reduction='batchmean',
                            log_target=True
                        )
                        loss = lm_loss + config.R_DROP_ALPHA * kl_loss + config.GENRE_LOSS_WEIGHT * genre_loss
                    else:
                        logits = outputs[0] if config.USE_GENRE_PREDICTION else outputs
                        vocab_size = logits.shape[-1]
                        if torch.isnan(logits).any() or torch.isinf(logits).any():
                            logging.warning("NaN or inf logits detected in training batch, skipping")
                            continue
                        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                        mask = target_ids != config.PAD_IDX
                        if mask.sum() > 0:
                            loss = (loss * mask.view(-1)).sum() / mask.sum()
                        else:
                            logging.warning("All tokens are padding, skipping batch")
                            continue
                        genre_loss = torch.tensor(0.0)
                        if config.USE_GENRE_PREDICTION and not genre_frozen:
                            genre_loss = genre_criterion(outputs[1], batch["genre_id"].to(config.DEVICE))
                            loss += config.GENRE_LOSS_WEIGHT * genre_loss

                    loss = loss / config.GRAD_ACCUM_STEPS

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN or inf loss detected, skipping batch")
                    continue

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_steps += 1

                if accumulated_steps % config.GRAD_ACCUM_STEPS == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0

                batch_loss = loss.item() * config.GRAD_ACCUM_STEPS
                total_train_loss += batch_loss
                if config.USE_GENRE_PREDICTION:
                    total_genre_loss += genre_loss.item()
                train_steps += 1

                batch_perplexity = math.exp(batch_loss) if batch_loss < 100 else float("inf")
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}", perplexity=f"{batch_perplexity:.2f}", accum=f"{accumulated_steps}/{config.GRAD_ACCUM_STEPS}")

            if accumulated_steps > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            avg_train_loss = total_train_loss / train_steps
            avg_genre_loss = total_genre_loss / train_steps if config.USE_GENRE_PREDICTION else 0.0

            val_metrics = evaluate(model, dev_loader, criterion, genre_criterion, config.DEVICE, vocab_size, tokenizer, config.PAD_IDX)
            val_loss = val_metrics["loss"]
            val_perplexity = val_metrics["perplexity"]
            val_rouge = val_metrics["rouge"]
            val_genre_loss = val_metrics["genre_loss"]

            current_lr = optimizer.param_groups[0]['lr']
            current_weight_decay = optimizer.param_groups[0].get('weight_decay', config.WEIGHT_DECAY)

            if config.USE_GENRE_PREDICTION and not genre_frozen:
                if val_genre_loss + config.GENRE_MIN_DELTA < best_genre_loss:
                    best_genre_loss = val_genre_loss
                    genre_epochs_no_improve = 0
                    logging.info(f"New best genre loss: {best_genre_loss:.4f}")
                    print(f"✅ New best genre loss: {best_genre_loss:.4f}")
                else:
                    genre_epochs_no_improve += 1
                    logging.info(f"Genre epochs without improvement: {genre_epochs_no_improve}")
                    print(f"🔍 Genre epochs without improvement: {genre_epochs_no_improve}")
                    if genre_epochs_no_improve >= config.GENRE_PATIENCE:
                        logging.info(f"Freezing genre head at epoch {epoch+1} due to no improvement in genre loss")
                        print(f"🛑 Freezing genre head at epoch {epoch+1}")
                        for param in model.genre_head.parameters():
                            param.requires_grad = False
                        genre_frozen = True

            epochs.append(epoch + 1)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            perplexities.append(val_perplexity)
            rouges.append(val_rouge)
            genre_losses.append(val_genre_loss)
            learning_rates.append(current_lr)
            weight_decays.append(current_weight_decay)

            logging.info(f"Epoch {epoch+1}/{config.EPOCHS}\n"
                         f"  Train Loss: {avg_train_loss:.4f}\n"
                         f"  Val Loss: {val_loss:.4f}\n"
                         f"  Val Perplexity: {val_perplexity:.2f}\n"
                         f"  ROUGE-L: {val_rouge:.4f}\n"
                         f"  Genre Loss: {val_genre_loss:.4f}\n"
                         f"  LR: {current_lr:.6f}\n"
                         f"  Mixed Precision: {config.MIXED_PRECISION}\n"
                         f"  Grad Accum Steps: {config.GRAD_ACCUM_STEPS}\n"
                         f"  Genre Frozen: {genre_frozen}")
            print(f"\nEpoch {epoch+1}/{config.EPOCHS}\n"
                  f"  Train Loss: {avg_train_loss:.4f}\n"
                  f"  Val Loss: {val_loss:.4f}\n"
                  f"  Val Perplexity: {val_perplexity:.2f}\n"
                  f"  ROUGE-L: {val_rouge:.4f}\n"
                  f"  Genre Loss: {val_genre_loss:.4f}\n"
                  f"  LR: {current_lr:.6f}\n"
                  f"  Mixed Precision: {config.MIXED_PRECISION}\n"
                  f"  Grad Accum Steps: {config.GRAD_ACCUM_STEPS}\n"
                  f"  Genre Frozen: {genre_frozen}")
            log_training_metrics(
                epoch + 1, avg_train_loss, val_loss, val_perplexity, val_rouge, val_genre_loss,
                os.path.join(config.STORAGE_DIR, "training_metrics.txt"),
                optimizer_name, current_lr, current_weight_decay
            )

            if scheduler:
                scheduler.step()
                logging.info(f"Scheduler epoch {epoch+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(
                    model, optimizer, epoch + 1, avg_train_loss, val_loss,
                    config.CHECKPOINT_DIR, scheduler, warmup_scheduler, scaler, is_best=True
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.PATIENCE:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % config.SAVE_DIS == 0:
                save_checkpoint(
                    model, optimizer, epoch + 1, avg_train_loss, val_loss,
                    config.CHECKPOINT_DIR, scheduler, warmup_scheduler, scaler
                )

            print(f"\n===== Stories for Epoch {epoch+1} =====")
            try:
                results = generate_multiple_stories(
                    model, tokenizer, config.QUERIES, label2id=label2id,
                    max_length=config.MAX_GEN_LEN,
                    min_length=50,
                    temperature=config.TEMPERATURE,
                    top_k=config.TOP_K,
                    top_p=config.TOP_P,
                    pad_idx=config.PAD_IDX,
                    eos_idx=config.EOS_IDX,
                    nhead=config.NHEAD
                )
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"Failed to generate stories for epoch {epoch+1}: {e}")
                results = []

            output_file = config.GENERATED_PATH
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"\n===== Stories for Epoch {epoch+1} =====\n")
            for i, (prompt, genre, story) in enumerate(results, 1):
                save_generated_story(story, prompt, i, output_file)

        except Exception as e:
            logging.error(f"Training loop failed at epoch {epoch+1}: {e}")
            raise

    final_model_path = os.path.join(config.CHECKPOINT_DIR, "transformer_final.pth")
    try:
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model: {final_model_path}")
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")

    print("\n===== Final Stories =====")
    try:
        final_results = generate_multiple_stories(
            model, tokenizer, config.QUERIES, label2id=label2id,
            max_length=config.MAX_GEN_LEN,
            min_length=50,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            top_p=config.TOP_P,
            pad_idx=config.PAD_IDX,
            eos_idx=config.EOS_IDX,
            nhead=config.NHEAD
        )
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Failed to generate final stories: {e}")
        final_results = []

    with open(config.GENERATED_PATH, "a", encoding="utf-8") as f:
        f.write("\n===== Final Stories =====\n")
    for i, (prompt, genre, story) in enumerate(final_results, 1):
        save_generated_story(story, prompt, i, config.GENERATED_PATH)

    plot_metrics(
        epochs,
        train_losses,
        val_losses,
        perplexities,
        rouges,
        genre_losses,
        learning_rates,
        weight_decays,
        os.path.join(config.STORAGE_DIR, "metrics_plot.png")
    )

if __name__ == "__main__":
    train()