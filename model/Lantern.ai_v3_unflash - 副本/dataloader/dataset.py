import torch
import logging
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import clean_genre_label, USE_GENRE_PREDICTION, LABEL2ID

class StoryDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_tokens=128, step=32, max_chunks=48):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_tokens = max_tokens
        self.step = step
        self.max_chunks = max_chunks
        logging.info(f"Using chunking params: max_tokens={self.max_tokens}, step={self.step}, max_chunks={self.max_chunks}")
        print(f"✅ Chunking params: max_tokens={self.max_tokens}, step={self.step}, max_chunks={self.max_chunks}")
        self.samples = self._tokenize_all()
        logging.info(f"Created dataset with {self.__len__()} chunks from {len(data)} stories")
        print(f"✅ Dataset created with {self.__len__()} chunks")

    def _tokenize_all(self):
        all_chunks = []
        total_chunks = 0
        skipped_stories = 0
        skip_reasons = {
            "invalid_structure": 0,
            "empty_text": 0,
            "no_tokens": 0,
            "no_chunks": 0,
            "missing_unk": 0,
            "missing_genre": 0
        }
        token_counts = []
        chunk_counts = []
        skipped_token_counts = []
        total_bytes = 0
        batch_texts = []
        batch_indices = []
        batch_size = 500

        for idx, story in enumerate(self.data):
            if not isinstance(story, dict) or "text" not in story or "labels" not in story:
                logging.warning(f"Skipping invalid story at index {idx}: missing 'text' or 'labels'")
                skip_reasons["invalid_structure"] += 1
                skipped_stories += 1
                continue

            text = story["text"].strip()
            if not text:
                logging.warning(f"Skipping empty story at index {idx}")
                skip_reasons["empty_text"] += 1
                skipped_stories += 1
                continue

            batch_texts.append(text)
            batch_indices.append(idx)
            
            if len(batch_texts) >= batch_size or idx == len(self.data) - 1:
                try:
                    encoded_batch = [self.tokenizer.encode(text, out_type=int) for text in batch_texts]
                except Exception as e:
                    logging.error(f"Failed to tokenize batch at index {idx}: {e}")
                    skip_reasons["no_tokens"] += len(batch_texts)
                    skipped_stories += len(batch_texts)
                    batch_texts, batch_indices = [], []
                    continue

                for text_idx, tokens, story_idx in zip(range(len(batch_texts)), encoded_batch, batch_indices):
                    tokens = [t if 0 <= t < self.tokenizer.get_vocab_size() else self.tokenizer.token_to_id("<unk>") for t in tokens]
                    if not tokens:
                        logging.warning(f"Skipping story at index {story_idx}: no tokens after encoding")
                        skip_reasons["no_tokens"] += 1
                        skipped_stories += 1
                        skipped_token_counts.append(0)
                        continue

                    primary_label = None
                    for label in self.data[story_idx]["labels"]:
                        cleaned_label = clean_genre_label(label)
                        if cleaned_label in self.label2id:
                            primary_label = cleaned_label
                            break
                    if not primary_label:
                        primary_label = "Unknown"
                        if story_idx % 5000 == 0:
                            logging.info(f"Assigned 'Unknown' genre for story at index {story_idx}")

                    label_id = self.label2id.get(primary_label, self.label2id.get("Unknown"))
                    if label_id is None:
                        logging.warning(f"Label ID not found for '{primary_label}' at index {story_idx}, using <unk>")
                        label_id = self.tokenizer.token_to_id("<unk>")
                        if label_id is None:
                            logging.error(f"<unk> token not found in tokenizer, skipping story {story_idx}")
                            skip_reasons["missing_unk"] += 1
                            skipped_stories += 1
                            skipped_token_counts.append(len(tokens))
                            continue

                    genre_token_id = self.tokenizer.token_to_id(f"<{primary_label}>")
                    if genre_token_id is None:
                        logging.warning(f"Genre token '<{primary_label}>' not found, using <Unknown> for story {story_idx}")
                        genre_token_id = self.tokenizer.token_to_id("<Unknown>")
                        if genre_token_id is None:
                            logging.error(f"<Unknown> token not found in tokenizer, skipping story {story_idx}")
                            skip_reasons["missing_genre"] += 1
                            skipped_stories += 1
                            skipped_token_counts.append(len(tokens))
                            continue

                    chunks = []
                    num_chunks = 0
                    for i in range(0, max(1, len(tokens) - self.max_tokens + 1), self.step):
                        chunk = tokens[i:i + self.max_tokens]
                        if len(chunk) < 1:
                            continue
                        chunk = [genre_token_id] + chunk
                        chunks.append({"input_ids": chunk, "genre_id": label_id})
                        num_chunks += 1
                        total_bytes += (len(chunk) * 4 + 4)
                        if num_chunks >= self.max_chunks:
                            break
                    if num_chunks > 0:
                        token_counts.append(len(tokens))
                        chunk_counts.append(num_chunks)
                        if story_idx % 5000 == 0:
                            logging.info(f"Sample {story_idx}: {num_chunks} chunks, {len(tokens)} tokens")
                    else:
                        logging.warning(f"Sample {story_idx} produced no chunks: {len(tokens)} tokens")
                        skip_reasons["no_chunks"] += 1
                        skipped_stories += 1
                        skipped_token_counts.append(len(tokens))
                    all_chunks.extend(chunks)
                    total_chunks += num_chunks

                batch_texts, batch_indices = [], []

        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            avg_chunks = sum(chunk_counts) / len(chunk_counts)
            dataset_size_mb = total_bytes / (1024 * 1024)
            logging.info(f"Token stats for used stories: avg={avg_tokens:.2f}, min={min_tokens}, max={max_tokens}")
            logging.info(f"Chunk stats: avg={avg_chunks:.2f}, total={total_chunks}")
            logging.info(f"Dataset size: {dataset_size_mb:.2f} MB")
            print(f"✅ Token stats for used stories: avg={avg_tokens:.2f}, min={min_tokens}, max={max_tokens}")
            print(f"✅ Chunk stats: avg={avg_chunks:.2f}, total={total_chunks}")
            print(f"✅ Dataset size: {dataset_size_mb:.2f} MB")
        if skipped_token_counts:
            avg_skipped_tokens = sum(skipped_token_counts) / len(skipped_token_counts) if skipped_token_counts else 0
            logging.info(f"Token stats for skipped stories: avg={avg_skipped_tokens:.2f}, count={skipped_stories}")
            print(f"✅ Token stats for skipped stories: avg={avg_skipped_tokens:.2f}, count={skipped_stories}")
        logging.info(f"Skip reasons: {skip_reasons}")
        print(f"✅ Skip reasons: {skip_reasons}")
        logging.info(f"Created dataset with {total_chunks} chunks from {len(self.data)} stories, skipped {skipped_stories} stories")
        print(f"✅ Dataset created with {total_chunks} chunks, skipped {skipped_stories} stories")
        return all_chunks

    def __len__(self):
        """Return the number of chunks in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        item = self.samples[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        sample = {
            "input_ids": input_ids[:-1],
            "target_ids": input_ids[1:],
            "genre_id": item["genre_id"]
        }
        return sample

def pad_sequences(sequences, padding_value=0):
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

def create_mask(input_ids, nhead):
    seq_length = input_ids.shape[1]
    mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.float, device=input_ids.device))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    batch_size = input_ids.shape[0]
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nhead, seq_length, seq_length)
    return mask.view(batch_size * nhead, seq_length, seq_length)

def create_padding_mask(input_ids, pad_idx=0):
    return input_ids.eq(pad_idx)

def collate_fn(batch, pad_idx=0, nhead=8):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_ids = pad_sequences(input_ids, padding_value=pad_idx)
    target_ids = pad_sequences(target_ids, padding_value=pad_idx)

    padding_mask = create_padding_mask(input_ids, pad_idx)
    attention_mask = create_mask(input_ids, nhead)

    result = {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "padding_mask": padding_mask
    }

    if USE_GENRE_PREDICTION:
        genre_ids = [torch.tensor(item["genre_id"], dtype=torch.long) for item in batch]
        result["genre_id"] = torch.stack(genre_ids)

    return result

def print_training_data_example(data_loader, tokenizer, pad_idx=0):
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

    if USE_GENRE_PREDICTION:
        print("\n📌 Genre IDs")
        print(batch["genre_id"])
        print(f"Shape: {batch['genre_id'].shape} (batch_size,)")

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