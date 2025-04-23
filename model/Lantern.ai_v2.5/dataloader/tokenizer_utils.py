import os
import logging
import sentencepiece as spm
from config import clean_genre_label  # Import to avoid duplication

def train_tokenizer(train_txt_path, tokenizer_prefix, vocab_size, genre_labels):
    """
    Train a SentencePiece tokenizer with genre-specific tokens.

    Args:
        train_txt_path (str): Path to the training text file.
        tokenizer_prefix (str): Prefix for tokenizer model and vocab files.
        vocab_size (int): Target vocabulary size.
        genre_labels (list): List of cleaned genre labels for special tokens.

    Raises:
        FileNotFoundError: If train_txt_path does not exist.
        RuntimeError: If tokenizer training fails.
    """
    model_path = f"{tokenizer_prefix}.model"
    if os.path.exists(model_path):
        logging.info(f"Tokenizer already exists at {model_path}, skipping training.")
        print(f"🔁 Tokenizer already exists at {model_path}")
        return

    if not os.path.exists(train_txt_path):
        logging.error(f"Training text file not found: {train_txt_path}")
        raise FileNotFoundError(f"Training text file not found: {train_txt_path}")

    logging.info("Starting SentencePiece tokenizer training...")
    print("🚀 Training SentencePiece tokenizer with genre labels...")

    # Create special tokens for genres (e.g., <Fantasy>, <Drama>)
    user_defined_symbols = ",".join([f"<{genre}>" for genre in genre_labels] + ["<Unknown>"])
    
    try:
        spm.SentencePieceTrainer.Train(
            input=train_txt_path,
            model_prefix=tokenizer_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type="bpe",
            user_defined_symbols=user_defined_symbols,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        logging.info(f"Tokenizer trained successfully and saved to {model_path}")
        print(f"✅ Tokenizer trained successfully: {model_path}")
    except Exception as e:
        logging.error(f"Failed to train tokenizer: {e}")
        print(f"❌ Error training tokenizer: {e}")
        raise RuntimeError(f"Tokenizer training failed: {e}")

def load_tokenizer(model_path):
    """
    Load a trained SentencePiece tokenizer.

    Args:
        model_path (str): Path to the tokenizer model file (.model).

    Returns:
        spm.SentencePieceProcessor: Loaded tokenizer instance.

    Raises:
        FileNotFoundError: If the model_path does not exist.
        RuntimeError: If loading the tokenizer fails.
    """
    if not os.path.exists(model_path):
        logging.error(f"Tokenizer model not found: {model_path}")
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(model_path)
        logging.info(f"Tokenizer loaded with vocab size: {tokenizer.get_piece_size()}")
        print(f"✅ Tokenizer loaded with vocab size: {tokenizer.get_piece_size()}")
        return tokenizer
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        print(f"❌ Error loading tokenizer: {e}")
        raise RuntimeError(f"Tokenizer loading failed: {e}")

def get_label2id(tokenizer, genre_labels):
    """
    Create a mapping from genre labels to their token IDs.

    Args:
        tokenizer (spm.SentencePieceProcessor): Loaded tokenizer.
        genre_labels (list): List of cleaned genre labels.

    Returns:
        dict: Mapping of genre names to their token IDs.

    Raises:
        ValueError: If any genre token is not found in the tokenizer's vocabulary.
    """
    label2id = {genre: tokenizer.piece_to_id(f"<{genre}>") for genre in genre_labels}
    
    # Verify all genre tokens exist
    missing_tokens = []
    for genre in genre_labels:
        token = f"<{genre}>"
        if tokenizer.piece_to_id(token) <= 0:  # 0 or negative means invalid
            missing_tokens.append(token)
    
    if missing_tokens:
        logging.error(f"Genre tokens not found in vocabulary: {missing_tokens}")
        raise ValueError(f"Genre tokens not found in vocabulary: {missing_tokens}")

    logging.info(f"Label2ID mapping created: {label2id}")
    print(f"✅ Label2ID mapping: {label2id}")
    return label2id

def extract_genre_labels(data):
    """
    Extract and clean unique genre labels from the dataset.

    Args:
        data (list): List of dataset entries with 'labels' field.

    Returns:
        list: Sorted list of cleaned genre labels.
    """
    genre_set = set()
    for item in data:
        for label in item.get("labels", []):
            cleaned_label = clean_genre_label(label)
            if cleaned_label:
                genre_set.add(cleaned_label)
    genre_labels = sorted(list(genre_set))
    logging.info(f"Found {len(genre_labels)} unique genres: {genre_labels}")
    print(f"✅ Found {len(genre_labels)} unique genres: {genre_labels}")
    return genre_labels