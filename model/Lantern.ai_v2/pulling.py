import torch
import os
import argparse
import logging
import json
from model.transformer import StoryTellerTransformer
from model.generate import generate_story
from dataloader.tokenizer_utils import load_tokenizer

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

def list_projects(storage_dir):
    """List all project directories in the storage folder."""
    if not os.path.exists(storage_dir):
        raise FileNotFoundError(f"Storage directory not found: {storage_dir}")
    
    projects = [d for d in os.listdir(storage_dir) 
                if os.path.isdir(os.path.join(storage_dir, d))]
    if not projects:
        raise FileNotFoundError("No project directories found in storage")
    
    return projects

def select_project(storage_dir):
    """Prompt user to select a project from the storage directory."""
    projects = list_projects(storage_dir)
    
    print("\n📂 Available projects:")
    for i, project in enumerate(projects, 1):
        print(f"{i}: {project}")
    
    while True:
        try:
            choice = input("\n🔍 Select project by number (1-{}): ".format(len(projects))).strip()
            choice = int(choice)
            if choice < 1 or choice > len(projects):
                print(f"⚠️ Invalid choice. Please enter a number between 1 and {len(projects)}.")
                continue
            selected_project = projects[choice - 1]
            project_dir = os.path.join(storage_dir, selected_project)
            logging.info(f"Selected project: {selected_project}")
            print(f"✅ Selected project: {selected_project}")
            return project_dir
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")

def load_model(checkpoint_dir, model_arch_path, tokenizer):
    """Load a model checkpoint based on user-selected file number."""
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

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoints found in checkpoints directory")

    print("\n📂 Available checkpoints:")
    for i, file in enumerate(checkpoint_files, 1):
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
            return model
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            logging.error(f"Checkpoint load error: {e}")
            print("Please choose again.")

def interactive_loop(model, tokenizer, label2id, min_len=100, max_length=250, pad_idx=0, eos_idx=3, nhead=8):
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
                pad_idx=pad_idx,
                eos_idx=eos_idx,
                nhead=nhead,
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

    # Get storage directory (same folder as pulling.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    storage_dir = os.path.join(script_dir, "storage")

    try:
        # Select project
        project_dir = select_project(storage_dir)
        
        # Set paths for the selected project
        tokenizer_path = os.path.join(project_dir, "movie_tokenizer.model")
        checkpoint_dir = os.path.join(project_dir, "checkpoints")
        model_arch_path = os.path.join(project_dir, "model_architecture.json")

        # Load tokenizer
        tokenizer = load_tokenizer(tokenizer_path)
        print(f"✅ Loaded tokenizer from {tokenizer_path}")

        # Get PAD_IDX and EOS_IDX from tokenizer
        pad_idx = tokenizer.piece_to_id("<pad>")
        eos_idx = tokenizer.piece_to_id("<|endoftext|>")
        if pad_idx is None or eos_idx is None:
            raise ValueError("Tokenizer missing <pad> or <|endoftext|> tokens")

        # Load model
        model = load_model(checkpoint_dir, model_arch_path, tokenizer)
        
        # Get nhead from architecture
        with open(model_arch_path, 'r', encoding='utf-8') as f:
            arch = json.load(f)
        nhead = arch["nhead"]

        # Start interactive loop
        interactive_loop(
            model=model,
            tokenizer=tokenizer,
            label2id=LABEL2ID,
            min_len=args.min_len,
            max_length=args.max_length,
            pad_idx=pad_idx,
            eos_idx=eos_idx,
            nhead=nhead
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Main error: {e}")
        raise

if __name__ == "__main__":
    main()