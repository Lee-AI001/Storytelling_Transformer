import torch
import os
import argparse
import logging
import json
from model.transformer import StoryTellerTransformer
from model.generate import generate_story, infer_genre
from dataloader.tokenizer_utils import load_tokenizer
from config import PAD_IDX, EOS_IDX, GENRES, LABEL2ID, USE_GENRE_PREDICTION, NUM_GENRES, DEVICE, STORAGE_DIR, PROJECT_NAME

# Configure logging for detailed debugging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
    logging.StreamHandler(),  # Output to console
    logging.FileHandler(os.path.join(STORAGE_DIR, "pulling.log"), mode='a', encoding='utf-8')  # Save to file
])

def detect_tokenizer_type(folder_path):
    """Detect tokenizer type based on files in the folder."""
    logging.debug(f"Checking tokenizer type in {folder_path}")
    model_path = os.path.join(folder_path, "story_tokenizer.model")
    if os.path.exists(model_path):
        logging.info(f"Found SentencePiece tokenizer at {model_path}")
        return "spm"
    logging.error(f"No tokenizer found at {model_path}")
    raise ValueError("Unable to detect tokenizer type: missing tokenizer files")

def load_model(model_arch_path, checkpoint_dir):
    """Load a model checkpoint based on user-selected folder and file number."""
    logging.debug(f"Loading model architecture from {model_arch_path}")
    try:
        with open(model_arch_path, 'r', encoding='utf-8') as f:
            arch = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load model architecture: {e}")
        raise FileNotFoundError(f"Model architecture file {model_arch_path} could not be loaded: {e}")

    global USE_GENRE_PREDICTION, NUM_GENRES
    USE_GENRE_PREDICTION = arch.get("use_genre_prediction", False)
    NUM_GENRES = len(GENRES) if USE_GENRE_PREDICTION else 0
    logging.debug(f"USE_GENRE_PREDICTION: {USE_GENRE_PREDICTION}, NUM_GENRES: {NUM_GENRES}")

    model = StoryTellerTransformer(
        vocab_size=arch["vocab_size"],
        d_model=arch["d_model"],
        nhead=arch["nhead"],
        num_layers=arch["num_layers"],
        dim_feedforward=arch.get("dim_feedforward", 0),
        dropout=arch.get("dropout", 0),
        embed_dropout=arch.get("embed_dropout", 0.1)
    )

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        logging.error(f"No checkpoints found in {checkpoint_dir}")
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    priority_files = ["transformer_best.pth", "transformer_final.pth"]
    available_priority = [f for f in priority_files if f in checkpoint_files]
    other_files = [f for f in checkpoint_files if f not in available_priority]
    sorted_files = available_priority + sorted(other_files)

    print("\n📂 Available checkpoints:")
    for i, file in enumerate(sorted_files, 1):
        print(f"{i}: {file}")
    
    while True:
        try:
            choice = input("\n🔍 Select checkpoint by number (1-{}): ".format(len(sorted_files))).strip()
            choice = int(choice)
            if choice < 1 or choice > len(sorted_files):
                print(f"⚠️ Invalid choice. Please enter a number between 1 and {len(sorted_files)}.")
                continue
            
            checkpoint_path = os.path.join(checkpoint_dir, sorted_files[choice - 1])
            logging.debug(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            model_state_dict = model.state_dict()
            checkpoint_state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
            checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_state_dict}
            model_state_dict.update(checkpoint_state_dict)
            model.load_state_dict(model_state_dict)
            
            genre_frozen = checkpoint.get("genre_frozen", False)
            if genre_frozen and USE_GENRE_PREDICTION:
                for param in model.genre_head.parameters():
                    param.requires_grad = False
                logging.info("Restored genre_frozen status: True")
            
            model.eval()
            model = model.to(DEVICE)
            logging.info(f"Successfully loaded checkpoint: {checkpoint_path}")
            return model, arch

        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except Exception as e:
            logging.error(f"Checkpoint load error: {e}")
            print(f"❌ Checkpoint load error: {e}")
            continue

def story_to_markdown(query, body, top_genre=None):
    """Convert story to Markdown format with user query and top predicted genre."""
    logging.debug(f"Formatting story to Markdown, query: {query[:50]}..., genre: {top_genre}")
    body = body.replace('\n', '\n\n')
    body = body.replace('--', '—')
    markdown_template = """User: 
{query}

{project_name}: 
<{genre}>
{body}
"""
    return markdown_template.format(query=query, project_name=PROJECT_NAME, genre=top_genre if top_genre else "None", body=body)

def interactive_loop(model, tokenizer, label2id, min_len=100, max_length=250, arch=None, selected_folder_path=None):
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

        output_file = os.path.join(selected_folder_path, "story_output.md")
        logging.debug(f"Preparing to write story to {output_file}")

        try:
            # Ensure selected_folder_path exists and is writable
            logging.debug(f"Checking selected folder: {selected_folder_path}")
            os.makedirs(selected_folder_path, exist_ok=True)
            if not os.access(selected_folder_path, os.W_OK):
                logging.error(f"No write permission for {selected_folder_path}")
                raise PermissionError(f"No write permission for {selected_folder_path}")

            # Infer genre if enabled
            top_genre = None
            if USE_GENRE_PREDICTION:
                logging.debug(f"Inferring genre for query: {query[:50]}...")
                top_genre, genre_probs = infer_genre(model, tokenizer, query, label2id)
                logging.info(f"Inferred top genre: {top_genre}, probabilities: {genre_probs}")

            # Generate story
            logging.debug(f"Generating story with max_length={max_len}, min_length={min_len}")
            with torch.amp.autocast('cuda', enabled=DEVICE.type == "cuda"):
                story = generate_story(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    label2id=label2id,
                    genre=None,
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
            logging.debug(f"Generated story: {story[:100]}...")

            # Validate story content
            if not story or story.strip() == "":
                logging.error("Generated story is empty or invalid")
                print("❌ Error: Generated story is empty")
                continue

            # Remove query from story if present
            query_lower = query.lower().strip()
            story_lower = story.lower()
            if story_lower.startswith(query_lower):
                story = story[len(query):].lstrip()
                logging.debug(f"Removed query prefix from story: {story[:100]}...")

            # Generate and write Markdown
            markdown_output = story_to_markdown(query, story, top_genre)
            logging.debug(f"Markdown output: {markdown_output[:100]}...")

            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_output)
                # Verify file creation
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logging.info(f"Successfully wrote story to {output_file} (Size: {os.path.getsize(output_file)/1024:.2f} KB)")
                    print(f"✅ Story saved to {output_file}")
                else:
                    logging.error(f"Failed to write story: {output_file} is empty or missing")
                    print(f"❌ Error: Story file {output_file} is empty or missing")
                    continue
            except IOError as e:
                logging.error(f"File write error for {output_file}: {e}")
                print(f"❌ Error writing to {output_file}: {e}")
                continue

            # Print story to console
            print("\n📜 Generated Story:")
            print("=" * 60)
            print(markdown_output)
            print("=" * 60)

        except Exception as e:
            logging.error(f"Error during story generation or file writing: {e}")
            print(f"❌ Error generating or saving story: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Interactive Story Generator")
    parser.add_argument("--min_len", type=int, default=100, help="Minimum story length in tokens")
    parser.add_argument("--max_length", type=int, default=250, help="Maximum story length in tokens")
    args = parser.parse_args()

    logging.debug(f"Starting main with STORAGE_DIR: {STORAGE_DIR}")
    storage_dir = os.path.join(os.path.dirname(__file__), "storage")
    if not os.path.exists(storage_dir):
        logging.error(f"Storage directory not found: {storage_dir}")
        raise FileNotFoundError("Storage directory not found")

    storage_folders = [f for f in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, f))]
    if not storage_folders:
        logging.error("No folders found in storage directory")
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
                logging.error(f"No 'checkpoints' subfolder found in {selected_folder}")
                raise FileNotFoundError(f"No 'checkpoints' subfolder found in {selected_folder}")
            break
        except ValueError:
            print("⚠️ Invalid input. Please enter a number.")
        except FileNotFoundError as e:
            logging.error(f"Folder selection error: {e}")
            print(f"⚠️ {e}")
            print("Please choose another folder.")

    try:
        tokenizer_path = os.path.join(selected_folder_path, "story_tokenizer.model")
        logging.debug(f"Loading tokenizer from {tokenizer_path}")
        if not os.path.exists(tokenizer_path):
            logging.error(f"Tokenizer file not found: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer file {tokenizer_path} not found")
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        raise FileNotFoundError(f"Failed to load tokenizer: {e}")

    try:
        model, arch = load_model(os.path.join(selected_folder_path, "model_architecture.json"), checkpoint_dir)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise FileNotFoundError(f"Failed to load model: {e}")

    interactive_loop(
        model=model,
        tokenizer=tokenizer,
        label2id=LABEL2ID,
        min_len=args.min_len,
        max_length=args.max_length,
        arch=arch,
        selected_folder_path=selected_folder_path
    )

if __name__ == "__main__":
    main()