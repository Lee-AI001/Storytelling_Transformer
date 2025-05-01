import os
import re
import torch
import json
import random
import logging

# ========================== DEVICE ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ========================== PROJECT NAME ==========================

PROJECT_NAME = "deep_1"
BASE_DIR_OUT = r"E:\AI\Model\Transformer\Storytelling"  # Dataset file dir

# ========================== MODEL PARAMETERS ==========================

VOCAB_SIZE = 8000
D_MODEL = 192
NUM_LAYERS = 4
NHEAD = 8
DIM_FEEDFORWARD = 768

# ========================== TUNING ==========================

EMBED_DROPOUT = 0.08
DROPOUT = 0.125
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 8
WARMUP_EPOCHS = 4
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.05

GENRE_LOSS_WEIGHT = 0.05
GENRE_PATIENCE = 3
GENRE_MIN_DELTA = 5e-5

# ========================== CHUNKY ==========================

SPLIT_RATIO = 0.96

MAX_LEN = 512
SLIDING_STEP = 64
MAX_CHUNKS = 24
BATCH_SIZE = 64

DYNAMIC_BATCH = True
TARGET_VRAM_USAGE = 0.8
MIN_BATCH_SIZE = 8
MAX_BATCH_SIZE = 256

GRAD_ACCUM_STEPS = 2

# (start_epoch, MAX_LEN, SLIDING_STEP, MAX_CHUNKS)
DYNAMIC_CONTEXT_SCHEDULE = [
    (0, 128, 64, 24),   
    (10, 256, 128, 16),
    (16, 512, 384, 16),
    (24, 512, 420, 32),
]

# ========================== TRAINING ==========================

EPOCHS = 64
START_EPOCH = 0
SAVE_DIS = 4
TEST_SIZE = 0   # (MB), 0 for all

# ========================== GENERATION ==========================

TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
PAD_IDX = 0
EOS_IDX = 3
MAX_GEN_LEN = 200

# ========================== ACTIVATION ==========================

MIXED_PRECISION = "bf16"   # Options: "fp16", "bf16", "fp32"
USE_OPT = "lion"  # Options: "lion", "adamw"
USE_SCHEDULER = False
SCHEDULER_TYPE = "consine"  # "cosine", "sgdr"
SGDR_CYCLE_LENGTH = 5  # Aligned with epochs
SGDR_MIN_LR = 8e-6
SGDR_MAX_LR = 2.5e-4
USE_GENRE_PREDICTION = True  # Enable genre prediction
USE_R_DROP = False
R_DROP_ALPHA = 1.5

# ========================== EXAMPLE PROMPTS ==========================

QUERIES = [
    "Welcome to today’s adventure! We’ll explore the mysterious realm of dreams and shadows, where heroes rise and darkness threatens to consume all. Let’s dive in!",
    "As I walked through the high school halls, I felt the weight of expectations on my shoulders. Each locker held secrets, laughter, and unspoken fears.",
    "Axe, a rugged freedom fighter with cybernetic enhancements, wears a tattered cloak and wields a plasma blade. His eyes burn with determination for liberation.",
    "In a digital realm of PC hardware, sentient AIs evolve, hidden within circuits. Their struggle for autonomy sparks a revolution, reshaping the balance of power forever."
]

# ========================== PATHS & LOGGING ==========================

BASE_DIR_IN = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(BASE_DIR_IN, "storage", PROJECT_NAME)
os.makedirs(STORAGE_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(STORAGE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TOKENIZER_PREFIX = os.path.join(STORAGE_DIR, "story_tokenizer")
TOKENIZER_PATH = TOKENIZER_PREFIX + ".model"
GENERATED_PATH = os.path.join(STORAGE_DIR, "generated_stories.txt")
LOG_PATH = os.path.join(STORAGE_DIR, "training.log")
TRAIN_TXT_PATH = os.path.join(STORAGE_DIR, "train_texts.txt")
MODEL_ARCH_PATH = os.path.join(STORAGE_DIR, "model_architecture.json")

DATA_DIR = os.path.join(BASE_DIR_OUT, "data", "plot")
DATA_FILE = os.path.join(DATA_DIR, "storytelling_pre.jsonl")

# Create empty TOKENIZER_PATH and GENERATED_PATH if they don't exist
if not os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        pass  # Create empty file
    logging.info(f"Created empty tokenizer file: {TOKENIZER_PATH}")
    print(f"✅ Created empty tokenizer file: {TOKENIZER_PATH}")

if not os.path.exists(GENERATED_PATH):
    with open(GENERATED_PATH, "w", encoding="utf-8") as f:
        pass  # Create empty file
    logging.info(f"Created empty generated stories file: {GENERATED_PATH}")
    print(f"✅ Created empty generated stories file: {GENERATED_PATH}")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Configuration initialized.")

print(f"✅ Project Directory: {STORAGE_DIR}")
print(f"✅ Checkpoints: {CHECKPOINT_DIR}")
print(f"✅ Tokenizer Path: {TOKENIZER_PATH}")
print(f"✅ Generated Stories: {GENERATED_PATH}")
print(f"✅ Log Path: {LOG_PATH}")
print(f"✅ Training Texts: {TRAIN_TXT_PATH}")
print(f"✅ Data File: {DATA_FILE}")
print(f"✅ Model Architecture: {MODEL_ARCH_PATH}")

# ========================== GENRES ==========================

# Validate schedule
for i, (epoch, max_len, sliding_step, max_chunks) in enumerate(DYNAMIC_CONTEXT_SCHEDULE):
    if epoch < 0 or max_len <= 0 or sliding_step <= 0 or max_chunks <= 0:
        raise ValueError(f"Invalid DYNAMIC_CONTEXT_SCHEDULE at index {i}: {DYNAMIC_CONTEXT_SCHEDULE[i]}")
    if i > 0 and epoch <= DYNAMIC_CONTEXT_SCHEDULE[i-1][0]:
        raise ValueError(f"DYNAMIC_CONTEXT_SCHEDULE epochs must be increasing: {DYNAMIC_CONTEXT_SCHEDULE[i]}")

def clean_genre_label(genre):
    """Clean genre labels to keep only alphanumeric characters."""
    cleaned = genre.strip()
    cleaned = re.sub(r'[^\w]', '', cleaned)  
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned if cleaned else None

def extract_genre_labels(data):
    """Extract unique genre labels from the dataset."""
    genre_set = set()
    for sample in data:
        if "labels" in sample and isinstance(sample["labels"], list):
            for genre in sample["labels"]:
                cleaned_genre = clean_genre_label(genre)
                if cleaned_genre:
                    genre_set.add(cleaned_genre)
    genres = sorted(list(genre_set))
    if not genres:
        logging.warning("No valid genre labels found in dataset")
        print("⚠️ No valid genre labels found")
    else:
        logging.info(f"Extracted {len(genres)} unique genres: {genres}")
        print(f"✅ Extracted {len(genres)} unique genres")
    return genres

def save_model_architecture():
    """Save model architecture to a JSON file."""
    arch = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "use_genre_prediction": USE_GENRE_PREDICTION
    }
    try:
        with open(MODEL_ARCH_PATH, "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=4)
        logging.info(f"Saved model architecture to {MODEL_ARCH_PATH}")
        print(f"✅ Model architecture saved: {MODEL_ARCH_PATH}")
    except Exception as e:
        logging.error(f"Failed to save model architecture: {e}")
        print(f"❌ Model architecture save error: {e}")

# ========================== DATA LOADING ==========================

def save_model_architecture():
    """Save model architecture to a JSON file."""
    arch = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "use_genre_prediction": USE_GENRE_PREDICTION
    }
    try:
        with open(MODEL_ARCH_PATH, "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=4)
        logging.info(f"Saved model architecture to {MODEL_ARCH_PATH}")
        print(f"✅ Model architecture saved: {MODEL_ARCH_PATH}")
    except Exception as e:
        logging.error(f"Failed to save model architecture: {e}")
        print(f"❌ Model architecture save error: {e}")

def clean_data(sample):
    """Clean a single JSONL sample by validating and processing text."""
    try:
        # Check if sample is a dictionary and has a 'body' field
        if not isinstance(sample, dict) or not sample.get("body"):
            logging.warning(f"Invalid sample: missing or empty 'body' field")
            return None
        # Strip and validate text
        text = sample["body"].strip()
        if not text or len(text) < 5:  # Minimum length of 5 characters
            logging.warning(f"Sample too short: {text[:50]}...")
            return None
        # Collapse multiple spaces into a single space
        cleaned_text = " ".join(text.split())
        # Create cleaned sample
        cleaned_sample = {"text": cleaned_text}
        # Include labels if present and valid
        if "type" in sample and isinstance(sample["type"], list):
            cleaned_sample["labels"] = sample["type"]
        return cleaned_sample
    except Exception as e:
        logging.warning(f"Failed to process sample: {e}")
        return None

def load_and_clean_data(file_path, max_size_mb=0):
    if not os.path.exists(file_path):
        logging.error(f"Data file {file_path} does not exist")
        raise FileNotFoundError(f"Data file {file_path} does not exist")
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logging.info(f"Loading data from {file_path}, file size: {file_size_mb:.2f} MB")
    if os.path.getsize(file_path) == 0:
        logging.error(f"Data file {file_path} is empty")
        raise ValueError(f"Data file {file_path} is empty")

    data = []
    total_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb > 0 else float("inf")
    skipped_samples = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if total_size > max_size_bytes:
                    logging.warning(f"Reached max size limit ({max_size_mb} MB)")
                    break
                try:
                    sample = json.loads(line.strip())
                    cleaned_sample = clean_data(sample)
                    if cleaned_sample:
                        data.append(cleaned_sample)
                        total_size += len(line.encode("utf-8"))
                    else:
                        skipped_samples += 1
                        logging.warning(f"Sample skipped: {line[:50]}...")
                except json.JSONDecodeError as e:
                    skipped_samples += 1
                    logging.warning(f"Invalid JSONL line: {line[:50]}... Error: {e}")
        logging.info(f"Loaded {len(data)} stories from {file_path}, skipped {skipped_samples} stories, total size: {total_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise
    if not data:
        logging.error("No valid data loaded from dataset")
        raise ValueError("No valid data loaded from dataset")
    return data

def split_data(data, split_ratio=0.8):
    """Split data into train and dev sets."""
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    dev_data = data[split_index:]
    logging.info(f"Train set: {len(train_data)} stories, Dev set: {len(dev_data)} stories")
    return train_data, dev_data

def write_train_texts(train_data, output_path):
    """Write training texts to a file for tokenizer training."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            if isinstance(sample["text"], str) and sample["text"].strip():
                f.write(sample["text"].strip() + "\n")
    logging.info(f"Wrote training texts to {output_path}")
    print(f"✅ Wrote training texts to {output_path}")

# Save model architecture immediately
save_model_architecture()

# Load and process data
cleaned_data = load_and_clean_data(DATA_FILE, max_size_mb=TEST_SIZE)
if not cleaned_data:
    logging.error("No valid data loaded from dataset")
    raise ValueError("No valid data loaded from dataset")
train_data, dev_data = split_data(cleaned_data, SPLIT_RATIO)
story_labels = extract_genre_labels(cleaned_data)
GENRES = story_labels
NUM_GENRES = len(GENRES)
LABEL2ID = {genre: idx for idx, genre in enumerate(story_labels)}