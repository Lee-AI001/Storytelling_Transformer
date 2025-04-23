import torch
import torch.nn.functional as F
import logging
import re
from config import clean_genre_label  # Import to avoid duplication
from .transformer import StoryTellerTransformer

# ========================== UTILITY FUNCTIONS ==========================

def create_causal_mask(seq_len, nhead, device, batch_size=1):
    """
    Create a causal attention mask for the transformer.

    Args:
        seq_len (int): Sequence length.
        nhead (int): Number of attention heads.
        device (torch.device): Device for tensor creation.
        batch_size (int, optional): Batch size. Defaults to 1.

    Returns:
        torch.Tensor: Causal mask of shape (batch_size * nhead, seq_len, seq_len).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float, device=device))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nhead, seq_len, seq_len)
    return mask.view(batch_size * nhead, seq_len, seq_len)

def top_p_filtering(logits, top_p):
    """
    Apply top-p (nucleus) filtering to logits.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, vocab_size).
        top_p (float): Cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: Filtered logits with low-probability tokens set to -inf.
    """
    batch_size, vocab_size = logits.shape
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask to keep valid indices
    mask = torch.ones_like(logits, dtype=torch.bool)
    for b in range(batch_size):
        mask[b, sorted_indices[b, sorted_indices_to_remove[b]]] = False

    return torch.where(mask, logits, torch.full_like(logits, float('-inf')))

# ========================== GENRE HANDLING ==========================

def infer_genre(query, label2id):
    query_lower = query.lower()
    genre_keywords = {
        "action": ["fight", "battle", "explosion", "chase", "hero"],
        "ai": ["artificial", "intelligence", "robot", "circuit", "algorithm"],
        "children": ["kid", "child", "fairy", "tale", "adventure"],
        "comedy": ["funny", "joke", "prank", "laugh", "humor"],
        "crime": ["detective", "murder", "heist", "thief", "criminal"],
        "drama": ["expectations", "secrets", "fears", "weight", "emotion"],
        "fantasy": ["dreams", "shadows", "heroes", "mage", "amulet"],
        "historical": ["war", "era", "king", "queen", "ancient"],
        "horror": ["ghost", "monster", "fear", "haunted", "darkness"],
        "nonfiction": ["true", "fact", "documentary", "history", "biography"],
        "other": ["unique", "miscellaneous", "experimental"],
        "realism": ["life", "everyday", "struggle", "reality"],
        "romance": ["love", "heart", "kiss", "passion", "relationship"],
        "science fiction": ["drones", "technology", "rebellion", "spaceship", "alien"],
        "speculative": ["what if", "future", "alternate", "possibility"],
        "thriller": ["suspense", "danger", "mystery", "stalker", "tension"],
        "young adult": ["teen", "school", "coming-of-age", "friendship", "youth"],
    }

    for genre, keywords in genre_keywords.items():
        cleaned_genre = clean_genre_label(genre)
        if cleaned_genre in label2id and any(kw in query_lower for kw in keywords):
            logging.info(f"Inferred genre '{cleaned_genre}' for query: {query[:50]}...")
            print(f"🔍 Inferred genre: {cleaned_genre}")
            return cleaned_genre
    logging.info(f"No genre inferred for query: {query[:50]}...")
    print("🔍 No genre inferred")
    return None

# ========================== STORY GENERATION ==========================

def generate_story(
    model, tokenizer, query, label2id=None, genre=None,
    max_length=250, min_length=50, temperature=0.9, pad_idx=0, eos_idx=3, top_k=50, top_p=0.9, nhead=8
):
    """
    Generate a single story from a query prompt.

    Args:
        model (StoryTellerTransformer): Trained transformer model.
        tokenizer (spm.SentencePieceProcessor): Trained tokenizer.
        query (str): Input prompt for story generation.
        label2id (dict, optional): Mapping of genres to token IDs.
        genre (str, optional): Specific genre to guide generation.
        max_length (int): Maximum length of generated story.
        min_length (int): Minimum length of generated story (excluding prompt and genre token).
        temperature (float): Sampling temperature for diversity.
        pad_idx (int): Padding token index.
        eos_idx (int): End-of-sequence token index.
        top_k (int): Number of top tokens for sampling.
        top_p (float): Cumulative probability for nucleus sampling.
        nhead (int): Number of attention heads in the model.

    Returns:
        str: Generated story text.
    """
    model.eval()
    device = next(model.parameters()).device

    # Validate vocab size
    tokenizer_vocab_size = tokenizer.get_piece_size()
    model_vocab_size = model.lm_head.out_features
    if tokenizer_vocab_size != model_vocab_size:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) != model vocab size ({model_vocab_size})")
        print(f"⚠️ Vocab size mismatch: tokenizer={tokenizer_vocab_size}, model={model_vocab_size}")

    # Infer genre if not provided
    if genre is None and label2id:
        genre = infer_genre(query, label2id)

    # Tokenize query
    try:
        tokens = tokenizer.encode(query, out_type=int)
    except Exception as e:
        logging.error(f"Failed to encode query '{query[:50]}...': {e}")
        print(f"❌ Encoding error: {e}")
        return "<Encoding failed>"

    # Add genre token
    if genre and label2id:
        genre = clean_genre_label(genre)
        genre_token_id = label2id.get(genre, tokenizer.piece_to_id("<Unknown>") if tokenizer.piece_to_id("<Unknown>") > 0 else 1)
        tokens = [genre_token_id] + tokens
        genre_token = f"<{genre}>"
    else:
        genre_token = None

    # Truncate if needed
    if len(tokens) >= max_length:
        logging.warning(f"Query tokens ({len(tokens)}) exceed max_length ({max_length}), truncating")
        print(f"⚠️ Truncating query to {max_length-1} tokens")
        tokens = tokens[:max_length-1]

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_ids.clone()
    min_reached = False
    prompt_len = input_ids.size(1)

    with torch.no_grad():
        while generated.shape[1] < max_length:
            seq_len = generated.shape[1]
            tgt_mask = create_causal_mask(seq_len, nhead, device)
            tgt_key_padding_mask = (generated == pad_idx)

            logits = model(generated, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            next_token_logits = logits[:, -1, :] / temperature

            # Handle NaN/inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.warning("NaN or inf in logits, clamping")
                print("⚠️ NaN/inf in logits")
                logits = torch.clamp(logits, min=-1e9, max=1e9)

            # Validate logits shape
            if next_token_logits.shape[1] != model_vocab_size:
                logging.error(f"Logits shape mismatch: expected [*, {model_vocab_size}], got {next_token_logits.shape}")
                print(f"❌ Logits shape error")
                break

            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Handle NaN/inf in probs
            if torch.isnan(next_token_probs).any() or torch.isinf(next_token_probs).any():
                logging.warning("NaN or inf in probabilities, using uniform distribution")
                print("⚠️ NaN/inf in probs")
                next_token_probs = torch.ones_like(next_token_probs) / model_vocab_size

            # Apply top-p or top-k filtering
            if top_p is not None:
                next_token_logits = top_p_filtering(next_token_logits, top_p)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
            elif top_k is not None:
                top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token)
            else:
                next_token = torch.multinomial(next_token_probs, num_samples=1)

            # Clamp token to valid range
            if next_token.max() >= model_vocab_size or next_token.min() < 0:
                logging.warning(f"Clamping out-of-range token: {next_token.item()}")
                print(f"⚠️ Clamping token: {next_token.item()}")
                next_token = torch.clamp(next_token, min=0, max=model_vocab_size-1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check if min_length is reached (exclude prompt and genre token)
            generated_len = generated.shape[1] - prompt_len
            if generated_len >= min_length:
                min_reached = True

            # Stop if EOS or pad and min_length met
            if next_token.item() in [pad_idx, eos_idx] and min_reached:
                break

            # If max_length reached but min_length not met, append EOS
            if generated.shape[1] == max_length and not min_reached:
                generated = torch.cat([generated, torch.tensor([[eos_idx]], device=device)], dim=1)
                break

    tokens = generated.squeeze(0).cpu().tolist()
    tokens = [t for t in tokens if 0 <= t < tokenizer_vocab_size]
    try:
        generated_text = tokenizer.decode(tokens)
    except Exception as e:
        logging.error(f"Failed to decode tokens: {e}")
        print(f"❌ Decoding error: {e}")
        generated_text = "<Decoding failed>"

    # Remove genre token from output
    if genre_token and generated_text.startswith(genre_token):
        generated_text = generated_text[len(genre_token):].strip()

    logging.info(f"Generated story for query '{query[:50]}...' (genre: {genre or 'None'}): {generated_text[:100]}...")
    print(f"✅ Story generated (genre: {genre or 'None'}): {generated_text[:50]}...")
    return generated_text


def generate_multiple_stories(
    model, tokenizer, queries, label2id=None, genres=None,
    max_length=250, temperature=0.9, pad_idx=0, eos_idx=3, top_k=50, top_p=0.9, nhead=8
):
    """
    Generate multiple stories from a list of query prompts.

    Args:
        model (StoryTellerTransformer): Trained transformer model.
        tokenizer (spm.SentencePieceProcessor): Trained tokenizer.
        queries (list): List of input prompts.
        label2id (dict, optional): Mapping of genres to token IDs.
        genres (list, optional): List of genres for each query.
        max_length (int): Maximum length of generated stories.
        temperature (float): Sampling temperature for diversity.
        pad_idx (int): Padding token index.
        eos_idx (int): End-of-sequence token index.
        top_k (int): Number of top tokens for sampling.
        top_p (float): Cumulative probability for nucleus sampling.
        nhead (int): Number of attention heads in the model.

    Returns:
        list: List of tuples (query, genre, generated_text).
    """
    model.eval()
    device = next(model.parameters()).device

    # Validate vocab size
    tokenizer_vocab_size = tokenizer.get_piece_size()
    model_vocab_size = model.lm_head.out_features
    if tokenizer_vocab_size != model_vocab_size:
        logging.warning(f"Tokenizer vocab size ({tokenizer_vocab_size}) != model vocab size ({model_vocab_size})")
        print(f"⚠️ Vocab size mismatch: tokenizer={tokenizer_vocab_size}, model={model_vocab_size}")

    # Prepare genres
    if genres is None:
        genres = [None] * len(queries)
    elif len(genres) != len(queries):
        logging.error(f"Genres length ({len(genres)}) != queries length ({len(queries)})")
        print(f"❌ Genre/query mismatch")
        raise ValueError("Number of genres must match number of queries")

    # Tokenize queries
    batch_tokens = []
    batch_genres = []
    for query, genre in zip(queries, genres):
        if not query.strip():
            logging.warning("Skipping empty query")
            print("⚠️ Skipping empty query")
            continue

        genre = genre or infer_genre(query, label2id) if label2id else None
        genre = clean_genre_label(genre) if genre else None

        try:
            tokens = tokenizer.encode(query, out_type=int)
        except Exception as e:
            logging.error(f"Failed to encode query '{query[:50]}...': {e}")
            print(f"❌ Encoding error for query: {e}")
            tokens = [tokenizer.unk_id()]

        if genre and label2id:
            genre_token_id = label2id.get(genre, tokenizer.piece_to_id("<Unknown>") if tokenizer.piece_to_id("<Unknown>") > 0 else 1)
            tokens = [genre_token_id] + tokens

        if len(tokens) >= max_length:
            logging.warning(f"Query tokens ({len(tokens)}) exceed max_length ({max_length}), truncating")
            print(f"⚠️ Truncating query to {max_length-1} tokens")
            tokens = tokens[:max_length-1]

        batch_tokens.append(tokens)
        batch_genres.append(genre)

    if not batch_tokens:
        logging.error("No valid queries provided")
        print("❌ No valid queries")
        return []

    # Pad sequences
    max_seq_len = min(max(len(t) for t in batch_tokens), max_length)
    input_ids = torch.full((len(batch_tokens), max_seq_len), pad_idx, dtype=torch.long, device=device)
    for i, tokens in enumerate(batch_tokens):
        input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)

    generated = input_ids.clone()
    batch_size = input_ids.shape[0]
    active = torch.ones(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(max_length - max_seq_len):
            seq_len = generated.shape[1]
            tgt_mask = create_causal_mask(seq_len, nhead, device, batch_size)
            tgt_key_padding_mask = (generated == pad_idx)

            logits = model(generated, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            next_token_logits = logits[:, -1, :] / temperature

            # Handle NaN/inf
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logging.warning("NaN or inf in logits, clamping")
                print("⚠️ NaN/inf in logits")
                logits = torch.clamp(logits, min=-1e9, max=1e9)

            # Validate logits shape
            if next_token_logits.shape[1] != model_vocab_size:
                logging.error(f"Logits shape mismatch: expected [*, {model_vocab_size}], got {next_token_logits.shape}")
                print(f"❌ Logits shape error")
                break

            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Handle NaN/inf in probs
            if torch.isnan(next_token_probs).any() or torch.isinf(next_token_probs).any():
                logging.warning("NaN or inf in probabilities, using uniform distribution")
                print("⚠️ NaN/inf in probs")
                next_token_probs = torch.ones_like(next_token_probs) / model_vocab_size

            # Apply top-p filtering
            if top_p is not None:
                next_token_logits = top_p_filtering(next_token_logits, top_p)
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
            elif top_k is not None:
                top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(top_k_probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token)
            else:
                next_token = torch.multinomial(next_token_probs, num_samples=1)

            # Clamp token to valid range
            if next_token.max() >= model_vocab_size or next_token.min() < 0:
                logging.warning(f"Clamping out-of-range tokens: {next_token.tolist()}")
                print(f"⚠️ Clamping tokens")
                next_token = torch.clamp(next_token, min=0, max=model_vocab_size-1)

            generated = torch.cat([generated, next_token], dim=1)

            done = (next_token.squeeze(-1) == pad_idx) | (next_token.squeeze(-1) == eos_idx)
            active = active & ~done
            if not active.any():
                break

    results = []
    for i, (query, genre) in enumerate(zip(queries, batch_genres)):
        tokens = generated[i].cpu().tolist()
        tokens = [t for t in tokens if 0 <= t < tokenizer_vocab_size]
        try:
            text = tokenizer.decode(tokens)
        except Exception as e:
            logging.error(f"Failed to decode tokens for query {i+1}: {e}")
            print(f"❌ Decoding error for query {i+1}: {e}")
            text = "<Decoding failed>"

        genre_token = f"<{genre}>" if genre else None
        if genre_token and text.startswith(genre_token):
            text = text[len(genre_token):].strip()

        results.append((query, genre, text))
        logging.info(f"Generated story {i+1} (genre: {genre or 'None'}): {text[:100]}...")
        print(f"✅ Story {i+1} generated (genre: {genre or 'None'}): {text[:50]}...")

    return results