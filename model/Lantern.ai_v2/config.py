import os
import torch
import json
import random
import logging


# ========================== PATHS & LOGGING ==========================

# Base project directory (absolute path to Storytelling folder)
BASE_DIR_OUT = r"E:\AI\Model\Transformer\Storytelling"                            # hardcode this path for Windows
BASE_DIR_IN = os.path.dirname(__file__)

# Storage directory for checkpoints, tokenizer, etc. (inside MHA_v2)
STORAGE_DIR = os.path.join(BASE_DIR_IN, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Specific paths
CHECKPOINT_DIR = os.path.join(STORAGE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
TOKENIZER_PREFIX = os.path.join(STORAGE_DIR, "movie_tokenizer")
TOKENIZER_PATH = TOKENIZER_PREFIX + ".model"
GENERATED_PATH = os.path.join(STORAGE_DIR, "generated_stories.txt")
LOG_PATH = os.path.join(STORAGE_DIR, "training.log")
TRAIN_TXT_PATH = os.path.join(STORAGE_DIR, "train_texts.txt")

# Data files (shared across projects)
DATA_DIR = os.path.join(BASE_DIR_OUT, "data", "plot")
DATA_FILE_1 = os.path.join(DATA_DIR, "movie_dataset.jsonl")
DATA_FILE_2 = os.path.join(DATA_DIR, "story_dataset.jsonl")
TRAIN_ON_STORY = True  # Combine both datasets if True, only movie if False

# Logging setup
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Configuration initialized.")

# Print paths for verification
print(f"✅ Storage Directory: {STORAGE_DIR}")
print(f"✅ Checkpoints: {CHECKPOINT_DIR}")
print(f"✅ Tokenizer Path: {TOKENIZER_PATH}")
print(f"✅ Generated Stories: {GENERATED_PATH}")
print(f"✅ Log Path: {LOG_PATH}")
print(f"✅ Training Texts: {TRAIN_TXT_PATH}")
print(f"✅ Data File 1: {DATA_FILE_1}")
print(f"✅ Data File 2: {DATA_FILE_2}")

# ========================== DEVICE ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# ========================== MODEL PARAMETERS ==========================

VOCAB_SIZE = 15000
D_MODEL = 160
NUM_LAYERS = 4
NHEAD = 10
DROPOUT = 0.275
DIM_FEEDFORWARD = 640

# ========================== TOKENIZATION ==========================

MAX_LEN = 512
SLIDING_STEP = 256
MAX_CHUNKS = 35

# ========================== TRAINING ==========================

BATCH_SIZE = 64
EPOCHS = 32
START_EPOCH = 28
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-1
PATIENCE = 3
WARMUP_EPOCHS = 2
MAX_GRAD_NORM = 1.0
USE_LION = False
USE_SCHEDULER = True
SCHEDULER_TYPE = "cosine"
SPLIT_RATIO = 0.975
SAVE_DIS = 2
TEST_SIZE = 0         # Number of samples to load for testing; 0 to load full dataset

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


# ========================== DATA LOADING ==========================

def load_and_clean_data(file_path):
    """Load and clean dataset, extracting text and genre labels."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Handle .jsonl format (one JSON object per line)
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        raise

    cleaned_data = []
    for story in raw_data:
        text = story.get("body", "").strip()
        if not text:
            continue
        # Expect "type" to be a list of genres
        labels = [str(lbl).strip() for lbl in story.get("type", []) if str(lbl).strip()]
        cleaned_data.append({"text": text, "labels": labels})
        
    logging.info(f"Cleaned {len(cleaned_data)} valid plots from {file_path}")
    print(f"✅ Cleaned {len(cleaned_data)} valid plots from {file_path}")
    return cleaned_data

def combine_datasets(file1, file2, test_size=0):
    """Combine two datasets if both exist, limit to test_size if specified."""
    data1 = load_and_clean_data(file1)
    if test_size > 0:
        data1 = data1[:test_size]
        logging.info(f"Limited dataset to {len(data1)} samples due to test_size={test_size}")
        print(f"✅ Limited dataset to {len(data1)} samples")
    if file2 and os.path.exists(file2):
        data2 = load_and_clean_data(file2)
        if test_size > 0:
            data2 = data2[:max(0, test_size - len(data1))]
            logging.info(f"Limited second dataset to {len(data2)} samples")
            print(f"✅ Limited second dataset to {len(data2)} samples")
        combined = data1 + data2
        logging.info(f"Combined datasets: {len(data1)} (file1) + {len(data2)} (file2) = {len(combined)} samples")
        print(f"✅ Combined datasets: {len(combined)} samples")
        return combined
    return data1

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
    # If genre is already in <genre> format, keep it
    if cleaned.startswith('<') and cleaned.endswith('>'):
        return cleaned[1:-1] if cleaned[2:-1] else None  # Extract inner genre (e.g., "realism")
    # Otherwise, clean invalid characters
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned if cleaned else None

# Load and process data
cleaned_data = combine_datasets(DATA_FILE_1, DATA_FILE_2 if TRAIN_ON_STORY else None, test_size=TEST_SIZE)
if not cleaned_data:
    logging.error("No valid data loaded from datasets")
    raise ValueError("No valid data loaded from datasets")
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