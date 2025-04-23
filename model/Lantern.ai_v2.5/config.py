import os
import torch
import json
import random
import logging

# ========================== DEVICE ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ========================== PROJECT NAME ==========================

PROJECT_NAME = "mid" 

# ========================== MODEL PARAMETERS ==========================

VOCAB_SIZE = 12000
D_MODEL = 192
NUM_LAYERS = 4
NHEAD = 12
DROPOUT = 0.225
DIM_FEEDFORWARD = 768

# ========================== TOKENIZATION ==========================

MAX_LEN = 256
SLIDING_STEP = 128
MAX_CHUNKS = 24

# ========================== TRAINING ==========================

BATCH_SIZE = 128
EPOCHS = 18
START_EPOCH = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1.5e-1
PATIENCE = 8
WARMUP_EPOCHS = 2
MAX_GRAD_NORM = 1.0
USE_LION = True
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"
SPLIT_RATIO = 0.95
SAVE_DIS = 6
TEST_SIZE = 0  # MB of data to load; 0 to load full dataset

# ========================== GENERATION ==========================

TEMPERATURE = 0.9
TOP_K = 50
TOP_P = 0.9
PAD_IDX = 0
EOS_IDX = 3
MAX_GEN_LEN = 128

# ========================== EXAMPLE PROMPTS ==========================

QUERIES = [
    "Welcome to today’s adventure! We’ll explore the mysterious realm of dreams and shadows, where heroes rise and darkness threatens to consume all. Let’s dive in!",
    "As I walked through the high school halls, I felt the weight of expectations on my shoulders. Each locker held secrets, laughter, and unspoken fears.",
    "In a world ravaged by technology, murder drones patrol the skies, hunting down the last survivors. A rebellion brews, and hope flickers in the darkness.",
    "As the storm raged, Captain Sparrow stood at the helm, eyes gleaming with mischief. 'Adventure awaits, mates! Let’s seize the treasure and defy the odds!'",
    "Axe, a rugged freedom fighter with cybernetic enhancements, wears a tattered cloak and wields a plasma blade. His eyes burn with determination for liberation.",
    "In a digital realm of PC hardware, sentient AIs evolve, hidden within circuits. Their struggle for autonomy sparks a revolution, reshaping the balance of power forever."
]

# ========================== PATHS & LOGGING ==========================

# Base project directory
BASE_DIR_OUT = r"E:\AI\Model\Transformer\Storytelling"
BASE_DIR_IN = os.path.dirname(__file__)

# Project-specific folder
STORAGE_DIR = os.path.join(BASE_DIR_IN, "storage", PROJECT_NAME)
os.makedirs(STORAGE_DIR, exist_ok=True)

# Specific paths
CHECKPOINT_DIR = os.path.join(STORAGE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TOKENIZER_PREFIX = os.path.join(STORAGE_DIR, "movie_tokenizer")
TOKENIZER_PATH = TOKENIZER_PREFIX + ".model"
GENERATED_PATH = os.path.join(STORAGE_DIR, "generated_stories.txt")
LOG_PATH = os.path.join(STORAGE_DIR, "training.log")
TRAIN_TXT_PATH = os.path.join(STORAGE_DIR, "train_texts.txt")
MODEL_ARCH_PATH = os.path.join(STORAGE_DIR, "model_architecture.json")  

# Data file
DATA_DIR = os.path.join(BASE_DIR_OUT, "data", "plot")
DATA_FILE = os.path.join(DATA_DIR, "storytelling_pre.jsonl")

# Logging setup
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Configuration initialized.")

# Print paths for verification
print(f"✅ Project Directory: {STORAGE_DIR}")
print(f"✅ Checkpoints: {CHECKPOINT_DIR}")
print(f"✅ Tokenizer Path: {TOKENIZER_PATH}")
print(f"✅ Generated Stories: {GENERATED_PATH}")
print(f"✅ Log Path: {LOG_PATH}")
print(f"✅ Training Texts: {TRAIN_TXT_PATH}")
print(f"✅ Data File: {DATA_FILE}")
print(f"✅ Model Architecture: {MODEL_ARCH_PATH}")

# ========================== DATA LOADING ==========================

# Save model architecture
def save_model_architecture():
    """Save model architecture to a JSON file."""
    arch = {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT
    }
    try:
        with open(MODEL_ARCH_PATH, "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=4)
        logging.info(f"Saved model architecture to {MODEL_ARCH_PATH}")
        print(f"✅ Model architecture saved: {MODEL_ARCH_PATH}")
    except Exception as e:
        logging.error(f"Failed to save model architecture: {e}")
        print(f"❌ Model architecture save error: {e}")

def load_and_clean_data(file_path, max_size_mb=0):
    """Load and clean dataset, extracting text and genre labels, limiting by size in MB."""
    try:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if max_size_mb > 0 and file_size > max_size_mb:
            logging.info(f"Limiting data to {max_size_mb} MB (file size: {file_size:.2f} MB)")
            print(f"📏 Limiting data to {max_size_mb} MB")
        else:
            max_size_mb = file_size
            logging.info(f"Loading full dataset ({file_size:.2f} MB)")
            print(f"📦 Loading full dataset ({file_size:.2f} MB)")

        data = []
        current_size = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                if current_size >= max_size_mb:
                    break
                raw = json.loads(line.strip())
                text = raw.get("body", "").strip()
                if not text:
                    continue
                labels = [str(lbl).strip() for lbl in raw.get("type", []) if str(lbl).strip()]
                data.append({"text": text, "labels": labels})
                current_size += len(line.encode('utf-8')) / (1024 * 1024)  # Size in MB

        logging.info(f"Loaded {len(data)} samples ({current_size:.2f} MB)")
        print(f"✅ Loaded {len(data)} samples ({current_size:.2f} MB)")
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise

def split_data(data, split_ratio=0.8):
    """Split data into train and dev sets."""
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    dev_data = data[split_index:]
    logging.info(f"Train set: {len(train_data)} samples, Dev set: {len(dev_data)} samples")
    print(f"✅ Train set: {len(train_data)} samples")
    print(f"✅ Dev set: {len(dev_data)} samples")
    return train_data, dev_data

def extract_genre_labels(data):
    """Extract and clean unique genre labels from the dataset."""
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

def clean_genre_label(genre):
    """Clean genre labels, preserving <genre> format if present."""
    import re
    cleaned = genre.strip()
    if cleaned.startswith('<') and cleaned.endswith('>'):
        return cleaned[1:-1] if cleaned[2:-1] else None
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned if cleaned else None

# Load and process data
cleaned_data = load_and_clean_data(DATA_FILE, max_size_mb=TEST_SIZE)
if not cleaned_data:
    logging.error("No valid data loaded from dataset")
    raise ValueError("No valid data loaded from dataset")
train_data, dev_data = split_data(cleaned_data, SPLIT_RATIO)
genre_labels = extract_genre_labels(cleaned_data)

# Write training texts for tokenizer
def write_train_texts(train_data, output_path):
    """Write training texts to a file for tokenizer training."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            if isinstance(sample["text"], str) and sample["text"].strip():
                f.write(sample["text"].strip() + "\n")
    logging.info(f"Wrote training texts to {output_path}")
    print(f"✅ Wrote training texts to {output_path}")

write_train_texts(train_data, TRAIN_TXT_PATH)
save_model_architecture()