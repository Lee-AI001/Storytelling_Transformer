import sentencepiece as spm
import os
import logging
import config

class UnifiedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, out_type=int):
        try:
            return self.tokenizer.encode(text, out_type=out_type)
        except Exception as e:
            logging.error(f"Failed to encode text: {e}")
            return []

    def decode(self, token_ids):
        try:
            return self.tokenizer.decode(token_ids)
        except Exception as e:
            logging.error(f"Failed to decode tokens: {e}")
            return ""

    def get_vocab_size(self):
        try:
            return self.tokenizer.get_piece_size()
        except Exception as e:
            logging.error(f"Failed to get vocab size: {e}")
            return 0

    def token_to_id(self, token):
        try:
            return self.tokenizer.piece_to_id(token)
        except Exception as e:
            logging.error(f"Failed to get token ID for {token}: {e}")
            return None

def train_tokenizer(input_path, output_prefix, vocab_size, story_labels):
    if not os.path.exists(input_path):
        logging.error(f"Input file {input_path} does not exist")
        raise FileNotFoundError(f"Input file {input_path} does not exist")
    
    if os.path.getsize(input_path) == 0:
        logging.error(f"Input file {input_path} is empty")
        raise ValueError(f"Input file {input_path} is empty")

    unique_labels = list(dict.fromkeys(story_labels))
    user_defined_symbols = [f"<{label}>" for label in unique_labels] + ["<Unknown>"]
    try:
        spm.SentencePieceTrainer.train(
            input=input_path,
            model_prefix=output_prefix,
            vocab_size=vocab_size,
            user_defined_symbols=user_defined_symbols,
            model_type="unigram",
            character_coverage=0.995,
            max_sentence_length=4096,
            shuffle_input_sentence=False
        )
        if not os.path.exists(config.TOKENIZER_PATH) or os.path.getsize(config.TOKENIZER_PATH) < 1024:
            logging.error(f"Tokenizer training failed: {config.TOKENIZER_PATH} missing or too small")
            raise RuntimeError(f"Tokenizer training failed: invalid output file {config.TOKENIZER_PATH}")
        logging.info(f"Trained SentencePiece tokenizer saved to {config.TOKENIZER_PATH}")
        print(f"✅ SentencePiece tokenizer trained: {config.TOKENIZER_PATH}")
    except Exception as e:
        logging.error(f"SentencePiece training failed: {e}")
        raise RuntimeError(f"SentencePiece training failed: {e}")

def load_tokenizer(model_path):
    if not os.path.exists(model_path):
        logging.error(f"Tokenizer file {model_path} does not exist")
        raise FileNotFoundError(f"Tokenizer file {model_path} does not exist")
    
    if os.path.getsize(model_path) < 1024:  # Assume <1KB is invalid
        logging.error(f"Tokenizer file {model_path} is too small or corrupted")
        raise ValueError(f"Tokenizer file {model_path} is invalid (size < 1KB)")
    
    try:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(model_path)
        if tokenizer.get_piece_size() == 0:
            logging.error(f"Tokenizer {model_path} has empty vocabulary")
            raise ValueError(f"Tokenizer {model_path} has empty vocabulary")
        logging.info(f"Loaded SentencePiece tokenizer from {model_path}")
        print(f"✅ Loaded SentencePiece tokenizer: {model_path}")
        return UnifiedTokenizer(tokenizer)
    except Exception as e:
        logging.error(f"Failed to parse SentencePiece model: {e}")
        raise RuntimeError(f"Failed to parse SentencePiece model {model_path}: {e}")

def get_label2id(tokenizer, labels):
    label2id = {}
    for label in labels:
        cleaned_label = config.clean_genre_label(label)
        if cleaned_label and cleaned_label in config.LABEL2ID:
            label2id[cleaned_label] = config.LABEL2ID[cleaned_label]
        else:
            logging.warning(f"Genre '{cleaned_label}' not found in LABEL2ID, mapping to <unk>")
            label2id[cleaned_label] = tokenizer.token_to_id("<unk>")
    return label2id