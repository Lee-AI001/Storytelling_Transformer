import torch
import torch.nn.functional as F
import logging
from config import DEVICE, LABEL2ID, USE_GENRE_PREDICTION, PAD_IDX, EOS_IDX

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    if torch.all(logits == filter_value):
        logging.warning("All logits filtered out, resetting to original logits")
        logits = logits.clone().fill_(0.0)
        logits[:, 0] = 1.0
    
    return logits

def infer_genre(model, tokenizer, query, label2id):
    if not USE_GENRE_PREDICTION:
        logging.warning("Genre prediction is disabled (USE_GENRE_PREDICTION=False)")
        print("⚠️ Genre prediction disabled")
        return None, []

    model.eval()
    try:
        encoded = tokenizer.encode(query, out_type=int)
        input_ids = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
        attention_mask = torch.ones_like(input_ids, dtype=torch.float).to(DEVICE)
        padding_mask = (input_ids == PAD_IDX).to(DEVICE)
        
        with torch.no_grad():
            _, genre_logits = model(input_ids, tgt_mask=None, tgt_key_padding_mask=padding_mask)
            probs = F.softmax(genre_logits, dim=-1)[0]
        
        id2label = {v: k for k, v in label2id.items()}
        top_k = 5
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        genre_probs = [(id2label.get(idx.item(), "Unknown"), prob.item()) 
                       for prob, idx in zip(top_probs, top_indices) if prob.item() > 0.05]
        
        predicted_idx = torch.argmax(probs, dim=-1).item()
        genre = id2label.get(predicted_idx, "Unknown")
        
        logging.info(f"Inferred top genre '{genre}' for query: {query[:50]}...")
        logging.info(f"Top {len(genre_probs)} genres: {genre_probs}")
        print(f"🔍 Inferred top genre: {genre}")
        print(f"🔍 Top genres: {genre_probs}")
        return genre, genre_probs
    except Exception as e:
        logging.error(f"Failed to infer genre: {e}")
        print(f"❌ Genre inference error: {e}")
        return None, []

def generate_story(model, tokenizer, query, label2id, genre=None, max_length=250, min_length=100, temperature=0.7, top_k=50, top_p=0.9, pad_idx=PAD_IDX, eos_idx=EOS_IDX, nhead=8):
    model.eval()
    try:
        genre, genre_probs = infer_genre(model, tokenizer, query, label2id)
        genre = genre or "Unknown"
        
        genre_token = f"<{genre}>"
        genre_id = label2id.get(genre, label2id.get("Unknown"))
        
        prompt = f"{genre_token} {query}"
        encoded = tokenizer.encode(prompt, out_type=int)
        if not encoded:
            logging.error(f"Empty encoding for prompt: {prompt[:50]}...")
            return "Error: Empty encoding"
        input_ids = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
        
        generated_ids = input_ids.clone()
        current_length = input_ids.size(1)
        past_tokens = set(encoded)
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                padding_mask = (generated_ids == pad_idx).to(DEVICE)
                
                with torch.amp.autocast('cuda'):  # Mixed precision
                    outputs = model(generated_ids, tgt_mask=None, tgt_key_padding_mask=padding_mask)
                if USE_GENRE_PREDICTION:
                    logits, _ = outputs
                else:
                    logits = outputs
                
                logits = logits[:, -1, :] / temperature
                logits = torch.clamp(logits, min=-1e4, max=1e4)  # Prevent NaNs
                
                for token in past_tokens:
                    logits[0, token] *= 0.8
                
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                
                if torch.all(logits == float('-inf')):
                    logging.warning(f"All logits filtered out for prompt: {query[:50]}...")
                    next_token = torch.tensor([[eos_idx]], dtype=torch.long, device=DEVICE)
                else:
                    probs = F.softmax(logits, dim=-1)
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        logging.warning(f"Invalid probabilities for prompt: {query[:50]}...")
                        next_token = torch.tensor([[eos_idx]], dtype=torch.long, device=DEVICE)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                past_tokens.add(next_token.item())
                current_length += 1
                
                if next_token.item() == eos_idx:
                    break
                
                if current_length >= min_length + input_ids.size(1) and torch.rand(1).item() < 0.05:
                    break
        
        decoded = tokenizer.decode(generated_ids[0].tolist())
        decoded = decoded.replace(f"<{genre}>", "").replace("<Unknown>", "").strip()
        decoded = ' '.join(decoded.split())
        
        if len(decoded.split()) < 10:
            logging.warning(f"Generated story too short: {decoded}")
            print("⚠️ Generated story too short")
            return "Story generation failed: too short."
        
        logging.info(f"Generated story for prompt: {query[:50]}...")
        return decoded
    except Exception as e:
        logging.error(f"Failed to generate story: {e}")
        print(f"❌ Story generation error: {e}")
        return f"Error generating story: {e}"

def generate_multiple_stories(model, tokenizer, queries, label2id, max_length=250, min_length=100, temperature=0.9, top_k=None, top_p=0.9, pad_idx=PAD_IDX, eos_idx=EOS_IDX, nhead=8):
    results = []
    for query in queries:
        try:
            if torch.cuda.is_available():
                logging.info(f"Before generating story for query '{query[:50]}...': {torch.cuda.memory_allocated(DEVICE) / 1024**3:.2f}GB")
            genre, genre_probs = infer_genre(model, tokenizer, query, label2id)
            story = generate_story(
                model, tokenizer, query, label2id, genre=genre,
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_idx=pad_idx,
                eos_idx=eos_idx,
                nhead=nhead
            )
            results.append((query, genre, story))
            logging.info(f"Generated story for query: {query[:50]}...")
            print(f"✅ Generated story for query: {query[:50]}...")
        except Exception as e:
            logging.error(f"Failed to generate story for query '{query[:50]}...': {e}")
            print(f"❌ Story generation error: {e}")
            results.append((query, None, f"Error: {e}"))
        finally:
            if torch.cuda.is_available():
                logging.info(f"After generating story for query '{query[:50]}...': {torch.cuda.memory_allocated(DEVICE) / 1024**3:.2f}GB")
    
    torch.cuda.empty_cache()
    return results