import torch
import os
import argparse
import logging
import json
from model.transformer import StoryTellerTransformer
from model.generate import generate_story, infer_genre
from dataloader.tokenizer_utils import load_tokenizer
from config import PAD_IDX, EOS_IDX, GENRES, LABEL2ID, USE_GENRE_PREDICTION, NUM_GENRES, DEVICE, STORAGE_DIR, PROJECT_NAME

def detect_tokenizer_type(folder_path):
    """Detect tokenizer type based on files in the folder."""
    model_path = os.path.join(folder_path, "story_tokenizer.model")
    if os.path.exists(model_path):
        return "spm"
    raise ValueError("Unable to detect tokenizer type: missing tokenizer files")

def load_model(model_arch_path, checkpoint_dir):
    """Load a model checkpoint based on user-selected folder and file number."""
    with open(model_arch_path, 'r', encoding='utf-8') as f:
        arch = json.load(f)
    
    global USE_GENRE_PREDICTION, NUM_GENRES
    USE_GENRE_PREDICTION = arch.get("use_genre_prediction", False)
    NUM_GENRES = len(GENRES) if USE_GENRE_PREDICTION else 0

    # Set defaults for dim_feedforward and dropout if not in arch
    model = StoryTellerTransformer(
        vocab_size=arch["vocab_size"],
        d_model=arch["d_model"],
        nhead=arch["nhead"],
        num_layers=arch["num_layers"],
        dim_feedforward=arch.get("dim_feedforward", 0),  # Default to 0
        dropout=arch.get("dropout", 0),  # Default to 0
        embed_dropout=arch.get("embed_dropout", 0.1)
    )

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

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
            
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
            checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict}
            model_state_dict.update(checkpoint_state_dict)
            model.load_state_dict(model_state_dict)
            
            model.eval()
            model = model.to(DEVICE)
            logging.info(f"Loaded checkpoint: {checkpoint_path}")
            return model, arch

        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except Exception as e:
            logging.error(f"Checkpoint load error: {e}")
            continue

def story_to_markdown(query, body, top_genre=None):
    """Convert story to Markdown format with user query and top predicted genre."""
    body = body.replace('\n', '\n\n')  # Double newlines for Markdown paragraphs
    body = body.replace('--', '—')  # Convert -- to em dash
    markdown_template = """User: 
{query}

{project_name}: 
<{genre}>
{body}
"""
    return markdown_template.format(query=query, project_name=PROJECT_NAME, genre=top_genre if top_genre else "None", body=body)

def interactive_loop(model, tokenizer, label2id, min_len=100, max_length=250, arch=None):
    print("🧠 Interactive Story Generator!")
    while True:
        query = input("\n📝 Enter your story prompt (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("👋 Exiting the story generator. See you next time!")
            break

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
            # Infer genre separately if genre prediction is enabled
            top_genre = None
            if USE_GENRE_PREDICTION:
                top_genre, _ = infer_genre(model, tokenizer, query, label2id)

            with torch.amp.autocast('cuda'):
                story = generate_story(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    label2id=label2id,
                    genre=None,  # Genre selection removed
                    max_length=max_len,
                    min_length=min_len,
                    temperature=temp,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    pad_idx=PAD_IDX,
                    eos_idx=EOS_IDX,
                    nhead=arch["nhead"],
                )
            torch.cuda.empty_cache()

            # Remove the query from the start of the story if present
            query_lower = query.lower().strip()
            story_lower = story.lower()
            if story_lower.startswith(query_lower):
                story = story[len(query):].lstrip()

        except Exception as e:
            print(f"⚠️ Error generating story: {e}")
            logging.error(f"Generation error: {e}")
            continue

        markdown_output = story_to_markdown(query, story, top_genre)
        output_file = os.path.join(STORAGE_DIR, "story_output.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)

        print("\n" + "=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Interactive Story Generator")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum story length in tokens")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum story length in tokens")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    if not os.path.exists(storage_dir):
        raise FileNotFoundError("Storage directory not found")

    storage_folders = [f for f in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, f))]
    if not storage_folders:
        raise FileNotFoundError("No folders found in storage directory")

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
            selected_folder_path = os.path.join(storage_dir, selected_folder)
            checkpoint_dir = os.path.join(selected_folder_path, "checkpoints")
            if not os.path.exists(checkpoint_dir):
                raise FileNotFoundError(f"No 'checkpoints' subfolder found in {selected_folder}")
            break
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except FileNotFoundError as e:
            print(f"⚠️ {e}")
            print("Please choose another folder.")

    try:
        tokenizer_path = os.path.join(selected_folder_path, "story_tokenizer.model")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found")
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load tokenizer: {e}")

    try:
        model, arch = load_model(os.path.join(selected_folder_path, "model_architecture.json"), checkpoint_dir)
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