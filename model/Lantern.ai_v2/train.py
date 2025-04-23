import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import math
from datetime import datetime
import re
from lion_pytorch import Lion
from model.transformer import StoryTellerTransformer
from dataloader.dataset import MoviePlotDataset, collate_fn, print_training_data_example
from dataloader.tokenizer_utils import load_tokenizer, get_label2id, train_tokenizer
from model.generate import generate_multiple_stories
import config
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import gc

# ========================== PLOT METRICS ==========================

def plot_metrics(epochs, train_losses, val_losses, perplexities, rouges, output_path):
    """Plot training and validation metrics and save to file."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, perplexities, label="Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.title("Perplexity")
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, rouges, label="ROUGE")
    plt.xlabel("Epoch")
    plt.ylabel("ROUGE")
    plt.legend()
    plt.title("ROUGE")
    
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved metrics plot to {output_path}")
        print(f"✅ Metrics plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save metrics plot: {e}")
        print(f"❌ Metrics plot save error: {e}")

# ========================== LOGGING METRICS ==========================

def log_training_metrics(epoch, train_loss, val_loss, val_perplexity, rouge, log_file, optimizer_name):
    """Log training metrics to a file."""
    log_entry = (
        f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
        f"Val Perplexity = {val_perplexity:.2f}, ROUGE = {rouge:.4f}, "
        f"Optimizer = {optimizer_name}"
    )
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        logging.info(f"Logged metrics: {log_entry}")
    except Exception as e:
        logging.error(f"Failed to log metrics: {e}")
        print(f"❌ Metrics logging error: {e}")

# ========================== SAVE GENERATED STORY ==========================

def save_generated_story(story, prompt, index, output_file):
    """Save generated story to a file."""
    try:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"Prompt {index}: {prompt[:100]}...\n")
            f.write(f"Story {index}: {story}\n\n")
        logging.info(f"Saved story {index} to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save story {index}: {e}")
        print(f"❌ Story save error: {e}")

# ========================== SAVE CHECKPOINT ==========================

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, scheduler=None, warmup_scheduler=None):
    """Save model checkpoint."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}_{timestamp}.pth")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if warmup_scheduler:
        checkpoint["warmup_scheduler_state_dict"] = warmup_scheduler.state_dict()
    
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")
        print(f"❌ Checkpoint save error: {e}")

# ========================== EVALUATION ==========================

def evaluate(model, data_loader, criterion, device, vocab_size, pad_idx=0, epoch=1):
    model.eval()
    total_loss = 0
    total_batches = 0
    references = []
    hypotheses = []

    if len(data_loader) == 0:
        logging.warning("Validation data loader is empty")
        print("⚠️ Empty validation data loader")
        return {"loss": float("inf"), "perplexity": float("inf"), "rouge": 0.0}

    logging.info("Starting evaluation...")
    print("📏 Starting evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating", leave=False)):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            logging.debug(f"Processing batch {batch_idx+1}/{len(data_loader)}")
            try:
                logits = model(input_ids, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask)

                if logits.shape[-1] != vocab_size:
                    logging.error(f"Logits shape mismatch: expected (*, {vocab_size}), got {logits.shape}")
                    print(f"❌ Logits shape error: {logits.shape}")
                    continue

                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                if pad_idx is not None:
                    mask = target_ids != pad_idx
                    if mask.sum() > 0:
                        loss = (loss * mask.view(-1)).sum() / mask.sum()
                    else:
                        logging.warning("All tokens are padding, skipping batch")
                        continue

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("NaN or inf loss detected, skipping batch")
                    print("⚠️ Invalid loss in batch")
                    continue

                total_loss += loss.item()
                total_batches += 1

                preds = torch.argmax(logits, dim=-1)
                for pred, ref in zip(preds.cpu().tolist(), target_ids.cpu().tolist()):
                    pred = [str(p) for p in pred if p != pad_idx]
                    ref = [str(r) for r in ref if r != pad_idx]
                    if pred and ref:
                        hypotheses.append(pred)
                        references.append([ref])

            except Exception as e:
                logging.error(f"Error in batch {batch_idx+1}: {e}")
                print(f"❌ Batch {batch_idx+1} error: {e}")
                continue

    if total_batches == 0:
        logging.warning("No valid batches processed in evaluation")
        print("⚠️ No valid evaluation batches")
        return {"loss": float("inf"), "perplexity": float("inf"), "rouge": 0.0}

    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    # Compute ROUGE
    rouge_score = 0.0
    if hypotheses:
        try:
            rouge_scorer_inst = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            rouge_scores = [rouge_scorer_inst.score(' '.join(ref[0]), ' '.join(hyp))['rougeL'].fmeasure
                           for ref, hyp in zip(references, hypotheses)]
            rouge_score = sum(rouge_scores) / len(rouge_scores)
            logging.info(f"Computed ROUGE score: {rouge_score:.4f}")
            print(f"✅ ROUGE score: {rouge_score:.4f}")
        except Exception as e:
            logging.error(f"ROUGE computation error: {e}")
            print(f"❌ ROUGE computation error: {e}")

    logging.info("Evaluation completed")
    print("✅ Evaluation completed")

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "rouge": rouge_score
    }

# ========================== TRAINING LOOP ==========================

def train():
    logging.info("Initializing training...")
    print("🚀 Starting training...")

    # Load tokenizer
    if not os.path.exists(config.TOKENIZER_PATH) or os.path.getsize(config.TOKENIZER_PATH) == 0:
        logging.info(f"Tokenizer missing or empty at {config.TOKENIZER_PATH}, training new tokenizer...")
        if not os.path.exists(config.TRAIN_TXT_PATH):
            logging.error(f"Training texts not found at {config.TRAIN_TXT_PATH}")
            raise FileNotFoundError(f"Training texts not found at {config.TRAIN_TXT_PATH}")
        train_tokenizer(config.TRAIN_TXT_PATH, config.TOKENIZER_PREFIX, config.VOCAB_SIZE, config.genre_labels)
    tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    label2id = get_label2id(tokenizer, config.genre_labels)

    tokenizer_vocab_size = tokenizer.get_piece_size()
    if tokenizer_vocab_size != config.VOCAB_SIZE:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) != config.VOCAB_SIZE ({config.VOCAB_SIZE})")
        print(f"⚠️ Vocab size mismatch: using {tokenizer_vocab_size}")
        vocab_size = tokenizer_vocab_size
    else:
        vocab_size = config.VOCAB_SIZE

    train_dataset = MoviePlotDataset(
        data=config.train_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_tokens=config.MAX_LEN,
        step=config.SLIDING_STEP,
        max_chunks=config.MAX_CHUNKS
    )
    dev_dataset = MoviePlotDataset(
        data=config.dev_data,
        tokenizer=tokenizer,
        label2id=label2id,
        max_tokens=config.MAX_LEN,
        step=config.SLIDING_STEP,
        max_chunks=config.MAX_CHUNKS
    )

    test_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=config.PAD_IDX, nhead=config.NHEAD)
    )
    print_training_data_example(test_loader, tokenizer, pad_idx=config.PAD_IDX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=config.PAD_IDX, nhead=config.NHEAD)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx=config.PAD_IDX, nhead=config.NHEAD)
    )

    if len(train_loader) == 0:
        logging.error("Training data loader is empty")
        print("❌ Empty training data loader")
        raise ValueError("Training data loader is empty")

    model = StoryTellerTransformer(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX, reduction="none")

    if config.USE_LION:
        optimizer = Lion(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        optimizer_name = "Lion"
        print("🦁 Using Lion optimizer")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        optimizer_name = "AdamW"
        print("🚀 Using AdamW optimizer")

    scheduler = None
    warmup_scheduler = None
    if config.USE_SCHEDULER:
        if config.SCHEDULER_TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.EPOCHS - config.WARMUP_EPOCHS,
                eta_min=config.LEARNING_RATE * 0.1
            )
            if config.WARMUP_EPOCHS > 0:
                warmup_lambda = lambda epoch: min((epoch + 1) / config.WARMUP_EPOCHS, 1.0)
                warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "transformer_best.pth")

    checkpoint_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith("transformer_epoch_") and f.endswith(".pth")]
    if checkpoint_files:
        try:
            latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(config.CHECKPOINT_DIR, x)))
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if config.USE_SCHEDULER and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if config.USE_SCHEDULER and config.SCHEDULER_TYPE == "cosine" and "warmup_scheduler_state_dict" in checkpoint:
                warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["val_loss"]
            logging.info(f"Resumed from checkpoint: {checkpoint_path} (epoch {start_epoch+1})")
            print(f"✅ Resumed from epoch {start_epoch+1} with val loss {best_val_loss:.4f}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            print(f"❌ Checkpoint load error: {e}. Starting from scratch.")

    # Collect metrics for plotting
    epochs = []
    train_losses = []
    val_losses = []
    perplexities = []
    rouges = []

    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
    
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(config.DEVICE)
            target_ids = batch["target_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            padding_mask = batch["padding_mask"].to(config.DEVICE)

            optimizer.zero_grad()
            output = model(input_ids, tgt_mask=attention_mask, tgt_key_padding_mask=padding_mask)

            if output.shape[-1] != vocab_size:
                logging.error(f"Output shape mismatch: expected (*, {vocab_size}), got {output.shape}")
                print(f"❌ Output shape error: {output.shape}")
                continue

            loss = criterion(output.view(-1, vocab_size), target_ids.view(-1))
            mask = target_ids != config.PAD_IDX
            if mask.sum() > 0:
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            else:
                logging.warning("All tokens are padding, skipping batch")
                print("⚠️ Skipping all-padding batch")
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN or inf loss detected, skipping batch")
                print("⚠️ Invalid loss in batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            perplexity = math.exp(batch_loss) if batch_loss < 100 else float("inf")
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", perplexity=f"{perplexity:.2f}")

        avg_train_loss = total_train_loss / len(train_loader)

        logging.info(f"Epoch {epoch+1} training completed, starting post-training steps")
        print(f"✅ Epoch {epoch+1} training completed")

        # Save checkpoint if validation loss improves
        logging.info("Checking for checkpoint save...")
        print("📥 Checking for checkpoint save...")
        try:
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model with val loss {best_val_loss:.4f}")
            print(f"✅ Saved best model: {best_model_path}")
        except Exception as e:
            logging.error(f"Failed to save best model: {e}")
            print(f"❌ Best model save error: {e}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                logging.info(f"Early stopping at epoch {epoch+1}")
                print("🛑 Early stopping triggered")
                break

        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_DIS == 0:
            logging.info("Saving periodic checkpoint...")
            print("📥 Saving periodic checkpoint...")
            try:
                save_checkpoint(
                    model, optimizer, epoch + 1, avg_train_loss, val_loss,
                    config.CHECKPOINT_DIR, scheduler, warmup_scheduler
                )
                logging.info(f"Saved periodic checkpoint for epoch {epoch+1}")
                print(f"✅ Saved periodic checkpoint")
            except Exception as e:
                logging.error(f"Failed to save periodic checkpoint: {e}")
                print(f"❌ Periodic checkpoint save error: {e}")

        # Generate stories
        logging.info(f"Generating stories for epoch {epoch+1}")
        print(f"📖 Generating stories for epoch {epoch+1}")
        import time
        start_time = time.time()
        try:
            results = generate_multiple_stories(
                model, tokenizer, config.QUERIES, label2id=label2id,
                max_length=config.MAX_GEN_LEN,
                temperature=config.TEMPERATURE,
                pad_idx=config.PAD_IDX,
                eos_idx=config.EOS_IDX,
                top_k=config.TOP_K,
                top_p=config.TOP_P,
                nhead=config.NHEAD
            )
            if time.time() - start_time > 300:
                logging.warning("Story generation took too long, proceeding to validation")
                print("⚠️ Story generation timeout")
            else:
                output_file = config.GENERATED_PATH
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"\n===== Stories for Epoch {epoch+1} =====\n")
                for i, (prompt, genre, story) in enumerate(results, 1):
                    print(f"\nPrompt {i}: {prompt[:100]}...")
                    print(f"Story {i}: {story[:250]}...")
                    save_generated_story(story, prompt, i, output_file)
                logging.info(f"Completed story generation for epoch {epoch+1}")
                print(f"✅ Completed story generation")
        except Exception as e:
            logging.error(f"Story generation error: {e}")
            print(f"❌ Story generation error: {e}")

        # Clear GPU memory
        logging.info("Clearing GPU memory...")
        print("🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()

        # Start validation
        logging.info(f"Starting validation for epoch {epoch+1}")
        print(f"🔍 Starting validation for epoch {epoch+1}")
        try:
            val_metrics = evaluate(model, dev_loader, criterion, config.DEVICE, vocab_size, config.PAD_IDX, epoch=epoch+1)
            val_loss = val_metrics["loss"]
            val_perplexity = val_metrics["perplexity"]
            val_rouge = val_metrics["rouge"]
        except Exception as e:
            logging.error(f"Validation error: {e}")
            print(f"❌ Validation error: {e}")
            val_loss = float("inf")
            val_perplexity = float("inf")
            val_rouge = 0.0

        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.2f}, ROUGE = {val_rouge:.4f}")
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")
        print(f"  ROUGE: {val_rouge:.4f}")
        log_training_metrics(
            epoch + 1, avg_train_loss, val_loss, val_perplexity, val_rouge,
            os.path.join(config.STORAGE_DIR, "training_metrics.txt"),
            optimizer_name
        )

        # Collect metrics
        epochs.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        perplexities.append(val_perplexity)
        rouges.append(val_rouge)

        if config.WARMUP_EPOCHS > 0 and epoch < config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        elif config.USE_SCHEDULER:
            scheduler.step()

    # Save final model
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "transformer_final.pth")
    try:
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model: {final_model_path}")
        print(f"✅ Final model saved: {final_model_path}")
    except Exception as e:
        logging.error(f"Failed to save final model: {e}")
        print(f"❌ Final model save error: {e}")

    # Generate final stories
    print("\n===== Final Stories =====")
    final_results = generate_multiple_stories(
        model, tokenizer, config.QUERIES, label2id=label2id,
        max_length=config.MAX_GEN_LEN,
        temperature=config.TEMPERATURE,
        pad_idx=config.PAD_IDX,
        eos_idx=config.EOS_IDX,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        nhead=config.NHEAD
    )

    with open(config.GENERATED_PATH, "a", encoding="utf-8") as f:
        f.write("\n===== Final Stories =====\n")
    for i, (prompt, genre, story) in enumerate(final_results, 1):
        print(f"\nPrompt {i}: {prompt[:100]}...")
        print(f"Story {i}: {story[:250]}...")
        save_generated_story(story, prompt, i, config.GENERATED_PATH)

    # Plot metrics
    plot_metrics(
        epochs,
        train_losses,
        val_losses,
        perplexities,
        rouges,
        os.path.join(config.STORAGE_DIR, "metrics_plot.png")
    )

if __name__ == "__main__":
    train()