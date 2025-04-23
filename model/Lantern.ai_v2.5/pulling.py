import torch
import os
import argparse
import logging
import json
from model.transformer import StoryTellerTransformer
from model.generate import generate_story
from dataloader.tokenizer_utils import load_tokenizer
from config import TOKENIZER_PATH, PAD_IDX, EOS_IDX

# Define genres (aligned with dataset)
GENRES = [
    "action",
    "ai",
    "children",
    "comedy",
    "crime",
    "drama",
    "fantasy",
    "historical",
    "horror",
    "nonfiction",
    "other",
    "realism",
    "romance",
    "science fiction",
    "speculative",
    "thriller",
    "young adult",
]

# Define label2id (matches tokenizer)
LABEL2ID = {genre: i + 4 for i, genre in enumerate(GENRES)}  # Start after special tokens

def load_model(model_arch_path):
    """Load a model checkpoint based on user-selected folder and file number."""
    with open(model_arch_path, 'r', encoding='utf-8') as f:
        arch = json.load(f)
    
    model = StoryTellerTransformer(
        vocab_size=arch["vocab_size"],
        d_model=arch["d_model"],
        nhead=arch["nhead"],
        num_layers=arch["num_layers"],
        dim_feedforward=arch["dim_feedforward"],
        dropout=arch["dropout"],
    )

    # Get storage directory (same directory as this script)
    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    if not os.path.exists(storage_dir):
        raise FileNotFoundError("Storage directory not found")

    # List all folders in storage
    storage_folders = [f for f in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, f))]
    if not storage_folders:
        raise FileNotFoundError("No folders found in storage directory")

    # List folders with numbers
    print("\n📂 Available folders in storage:")
    for i, folder in enumerate(sorted(storage_folders), 1):
        print(f"{i}: {folder}")
    
    while True:
        try:
            folder_choice = input("\n🔍 Select folder by number (1-{}): ".format(len(storage_folders))).strip()
            folder_choice = int(folder_choice)
            if folder_choice < 1 or folder_choice > len(storage_folders):
                print(f"⚠️ Invalid choice. Please enter a number between 1 and {len(storage_folders)}.")
                continue
            selected_folder = storage_folders[folder_choice - 1]
            checkpoint_dir = os.path.join(storage_dir, selected_folder, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f"No 'checkpoints' subfolder found in {selected_folder}")
            break
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            print("Please choose another folder.")
    
    # Get all checkpoint files in checkpoints subfolder
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # List checkpoints with numbers
    print("\n📂 Available checkpoints:")
    for i, file in enumerate(sorted(checkpoint_files), 1):
        print(f"{i}: {file}")
    
    while True:
        try:
            choice = input("\n🔍 Select checkpoint by number (1-{}): ".format(len(checkpoint_files))).strip()
            choice = int(choice)
            if choice < 1 or choice > len(checkpoint_files):
                print(f"⚠️ Invalid choice. Please enter a number between 1 and {len(checkpoint_files)}.")
                continue
            
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[choice - 1])
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            logging.info(f"Loaded checkpoint: {checkpoint_path}")
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
            return model, arch

        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            logging.error(f"Checkpoint load error: {e}")
            print("Please choose again.")

def interactive_loop(model, tokenizer, label2id, min_len=100, max_length=250, arch=None):
    print("🧠 Interactive Story Generator!")
    print("Available genres:")
    for i, genre in enumerate(GENRES, 1):
        print(f"{i}. {genre}")
    print("0. none (default)")

    while True:
        query = input("\n📝 Enter your story prompt (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("👋 Exiting the story generator. See you next time!")
            break

        try:
            genre_idx = input("🎭 Genre (0 for none, 1-17 for genres, default=0): ") or "0"
            genre_idx = int(genre_idx)
            if genre_idx < 0 or genre_idx > 17:
                raise ValueError("Genre index out of range")
            genre = None if genre_idx == 0 else GENRES[genre_idx - 1]
        except ValueError:
            print("⚠️ Invalid genre, using none.")
            genre = None

        use_custom = input("🔧 Tune hyperparameters? (yes/no, default=no): ").lower()
        if use_custom in ["yes", "y"]:
            try:
                temp = float(input("🌡️ Temperature (default=0.9): ") or 0.9)
                top_k = int(input("🎯 Top-k (default=50, 0 to disable): ") or 50)
                top_p = float(input("🎲 Top-p (default=0.9): ") or 0.9)
                max_len = int(input("📏 Max length (default=250): ") or 250)
                min_len = int(input("📐 Min length (default=100): ") or 100)
                if min_len >= max_len:
                    print("⚠️ Min length must be less than max length, using defaults.")
                    max_len = 250
                    min_len = 100
            except ValueError:
                print("⚠️ Invalid input, using default settings.")
                temp = 0.9
                top_k = 50
                top_p = 0.9
                max_len = 250
                min_len = 100
        else:
            temp = 0.9
            top_k = 50
            top_p = 0.9
            max_len = 250
            min_len = 100
            print("✅ Using default settings: temperature=0.9, top_k=50, top_p=0.9, max_length=250, min_length=100")

        try:
            story = generate_story(
                model=model,
                tokenizer=tokenizer,
                query=query,
                label2id=label2id,
                genre=genre,
                max_length=max_len,
                min_length=min_len,
                temperature=temp,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                pad_idx=PAD_IDX,
                eos_idx=EOS_IDX,
                nhead=arch["nhead"],
            )
        except Exception as e:
            print(f"⚠️ Error generating story: {e}")
            logging.error(f"Generation error: {e}")
            continue

        print("\n📖 Generated Story:")
        print(story)
        print("\n" + "=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Interactive Story Generator")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum story length in tokens")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum story length in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        print(f"✅ Loaded tokenizer from {TOKENIZER_PATH}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to load tokenizer: {e}")

    try:
        model, arch = load_model(
            os.path.join(os.path.dirname(TOKENIZER_PATH), "model_architecture.json")
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model: {e}")

    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        label2id=LABEL2ID,
        min_len=args.min_len,
        max_length=args.max_length,
        arch=arch
    )

if __name__ == "__main__":
    main()