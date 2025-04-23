import torch
import logging
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import clean_genre_label  # Import to avoid duplication

class MoviePlotDataset(Dataset):
    """
    Dataset for movie plots, tokenizing text and chunking with sliding windows.

    Args:
        data (list): List of dicts with 'text' and 'labels' keys.
        tokenizer (spm.SentencePieceProcessor): Trained SentencePiece tokenizer.
        label2id (dict): Mapping of genre labels to token IDs.
        max_tokens (int): Maximum tokens per chunk.
        step (int): Sliding window step size.
        max_chunks (int): Maximum chunks per story.
    """
    def __init__(self, data, tokenizer, label2id, max_tokens=512, step=400, max_chunks=12):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_tokens = max_tokens
        self.step = step
        self.max_chunks = max_chunks
        self.samples = self._tokenize_all()
        logging.info(f"Created dataset with {len(self.samples)} chunks from {len(data)} stories")
        print(f"✅ Dataset created with {len(self.samples)} chunks")

    def _tokenize_all(self):
        all_chunks = []
        for idx, story in enumerate(self.data):
            # Validate story
            if not isinstance(story, dict) or "text" not in story or "labels" not in story:
                logging.warning(f"Skipping invalid story at index {idx}: missing 'text' or 'labels'")
                continue

            text = story["text"].strip()
            if not text:
                logging.warning(f"Skipping empty story at index {idx}")
                continue

            # Tokenize text
            try:
                tokens = self.tokenizer.encode(text, out_type=int)
            except Exception as e:
                logging.error(f"Failed to tokenize story at index {idx}: {e}")
                continue

            # Select primary genre label
            primary_label = None
            for label in story["labels"]:
                cleaned_label = clean_genre_label(label)
                if cleaned_label in self.label2id:
                    primary_label = cleaned_label
                    break
            primary_label = primary_label or "Unknown"
            label_id = self.label2id.get(primary_label)
            if label_id is None:
                logging.warning(f"Unknown genre '{primary_label}' for story at index {idx}, using default")
                label_id = self.tokenizer.piece_to_id("<Unknown>") if self.tokenizer.piece_to_id("<Unknown>") > 0 else 1  # Fallback to unk_id

            # Sliding window chunking
            chunks = []
            for i in range(0, len(tokens), self.step):
                chunk = tokens[i:i + self.max_tokens]
                if len(chunk) < 2:  # Skip tiny chunks
                    continue
                chunk = [label_id] + chunk  # Prepend genre token
                chunks.append({"input_ids": chunk})
                if len(chunks) >= self.max_chunks:
                    break
            all_chunks.extend(chunks)

        return all_chunks

    def __len__(self):
        """Return the number of chunks in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary with 'input_ids' and 'target_ids' (shifted by 1).
        """
        item = self.samples[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        return {
            "input_ids": input_ids[:-1],  # Input for prediction
            "target_ids": input_ids[1:]   # Target for loss
        }

def pad_sequences(sequences, padding_value=0):
    """
    Pad a list of sequences to the same length.

    Args:
        sequences (list): List of torch tensors.
        padding_value (int): Value to use for padding.

    Returns:
        torch.Tensor: Padded tensor with shape (batch_size, max_len).
    """
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

def create_mask(input_ids, nhead):
    """
    Create a causal attention mask for the transformer.

    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        nhead (int): Number of attention heads.

    Returns:
        torch.Tensor: Causal mask of shape (batch_size * nhead, seq_len, seq_len).
    """
    seq_length = input_ids.shape[1]
    mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.float, device=input_ids.device))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    batch_size = input_ids.shape[0]
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nhead, seq_length, seq_length)
    return mask.view(batch_size * nhead, seq_length, seq_length)

def create_padding_mask(input_ids, pad_idx=0):
    """
    Create a padding mask for the input tensor.

    Args:
        input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        pad_idx (int): Index of the padding token.

    Returns:
        torch.Tensor: Padding mask of shape (batch_size, seq_len).
    """
    return input_ids.eq(pad_idx)

def collate_fn(batch, pad_idx=0, nhead=8):
    """
    Collate a batch of dataset samples, padding and creating masks.

    Args:
        batch (list): List of dataset samples.
        pad_idx (int): Index of the padding token.
        nhead (int): Number of attention heads.

    Returns:
        dict: Dictionary with padded 'input_ids', 'target_ids', 'attention_mask', and 'padding_mask'.
    """
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_ids = pad_sequences(input_ids, padding_value=pad_idx)
    target_ids = pad_sequences(target_ids, padding_value=pad_idx)

    padding_mask = create_padding_mask(input_ids, pad_idx)
    attention_mask = create_mask(input_ids, nhead)

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "padding_mask": padding_mask
    }

def print_training_data_example(data_loader, tokenizer, pad_idx=0):
    """
    Print a sample batch from the data loader for debugging.

    Args:
        data_loader (DataLoader): PyTorch DataLoader instance.
        tokenizer (spm.SentencePieceProcessor): Tokenizer for decoding.
        pad_idx (int): Index of the padding token.
    """
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        logging.error("DataLoader is empty")
        print("❌ DataLoader is empty!")
        return

    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    attention_mask = batch["attention_mask"]
    padding_mask = batch["padding_mask"]

    print("=" * 60)
    print("📌 Input IDs (Padded Tokens)")
    print(input_ids)
    print(f"Shape: {input_ids.shape} (batch_size, seq_len)")

    print("\n📌 Target IDs")
    print(target_ids)
    print(f"Shape: {target_ids.shape}")

    print("\n📌 Causal Attention Mask (First Head)")
    print(attention_mask[0])
    print(f"Shape: {attention_mask.shape} (batch_size * nhead, seq_len, seq_len)")

    print("\n📌 Padding Mask")
    print(padding_mask)
    print(f"Shape: {padding_mask.shape} (batch_size, seq_len)")

    # Decode a sample sequence
    sample_idx = 0
    sample_input = input_ids[sample_idx].tolist()
    try:
        decoded_text = tokenizer.decode(sample_input)
        print("\n📌 Decoded Sample Input")
        print(decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text)
    except Exception as e:
        logging.error(f"Failed to decode sample input: {e}")
        print(f"❌ Failed to decode sample: {e}")

    logging.info("Validated data loader sample")
    print("✅ Data loader sample validated!")