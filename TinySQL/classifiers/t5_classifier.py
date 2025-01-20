from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm


class TextLabelDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = self.tokenizer(
            str(label),
            max_length=2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
        }


def train_t5_classifier(data, seed=42, max_epochs=3, batch_size=8, lr=1e-4, model=None):
    # Set seed for reproducibility
    torch.manual_seed(seed)

    model_name = "t5-large"
    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
    train_dataset = TextLabelDataset(train_data, tokenizer)
    test_dataset = TextLabelDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if model is None:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        # Create datasets and dataloaders
        # Set up optimizer
        optimizer = AdamW(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(max_epochs):
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
            predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            true_labels = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

            correct += sum(p == t for p, t in zip(predictions, true_labels))
            total += len(predictions)

    accuracy = correct / total
    return accuracy, model
