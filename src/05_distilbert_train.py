"""
DistilBERT Fine-tuning for RAM Sentiment Analysis
Model: distilbert-base-multilingual-cased
Dataset: 1,664 reviews (3-class: Negative, Neutral, Positive)
"""
import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = '/home/ubuntu/ram_thesis_project/outputs/merged_labeled_complete.csv'
OUT_DIR     = '/home/ubuntu/ram_thesis_project/outputs/distilbert'
VIZ_DIR     = '/home/ubuntu/ram_thesis_project/outputs/visualizations'
MODEL_NAME  = 'distilbert-base-multilingual-cased'
MAX_LEN     = 64
BATCH_SIZE  = 8
EPOCHS      = 3
LR          = 2e-5
SEED        = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

LABEL_MAP   = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
ID2LABEL    = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Columns: {df.columns.tolist()}")
df = df.dropna(subset=['review_text', 'sentiment_3'])
df = df[df['review_text'].str.len() > 5]
df['label'] = df['sentiment_3'].map(LABEL_MAP)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"Total samples: {len(df)}")
print(df['sentiment_3'].value_counts())

# ── Split ─────────────────────────────────────────────────────────────────────
X = df['review_text'].tolist()
y = df['label'].tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ── Class weights ─────────────────────────────────────────────────────────────
classes = np.array([0, 1, 2])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float)
print(f"Class weights: {weights}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=MAX_LEN, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]

print("Tokenizing datasets...")
train_ds = ReviewDataset(X_train, y_train)
val_ds   = ReviewDataset(X_val,   y_val)
test_ds  = ReviewDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3)
model.to(device)
class_weights = class_weights.to(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ── Training loop ─────────────────────────────────────────────────────────────
train_losses, val_losses, val_accs = [], [], []

def evaluate(loader):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels

best_val_loss = float('inf')
best_model_path = os.path.join(OUT_DIR, 'best_model.pt')

print(f"\nStarting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for step, (batch, labels) in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
        if (step + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = epoch_loss / len(train_loader)
    val_loss, val_acc, _, _ = evaluate(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> Best model saved (val_loss={val_loss:.4f})")

# ── Final evaluation on test set ──────────────────────────────────────────────
print("\nLoading best model for final evaluation...")
model.load_state_dict(torch.load(best_model_path, map_location=device))
test_loss, test_acc, test_preds, test_labels = evaluate(test_loader)

report = classification_report(
    test_labels, test_preds,
    target_names=['Negative', 'Neutral', 'Positive'],
    output_dict=True)
macro_f1 = f1_score(test_labels, test_preds, average='macro')
weighted_f1 = f1_score(test_labels, test_preds, average='weighted')

print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS")
print(f"{'='*60}")
print(f"Accuracy:    {test_acc*100:.2f}%")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(classification_report(test_labels, test_preds,
      target_names=['Negative', 'Neutral', 'Positive']))

# ── Save results JSON ─────────────────────────────────────────────────────────
results = {
    'model': MODEL_NAME,
    'dataset_size': len(df),
    'train_size': len(X_train),
    'val_size': len(X_val),
    'test_size': len(X_test),
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'max_len': MAX_LEN,
    'class_weights': {ID2LABEL[i]: round(float(weights[i]), 4) for i in range(3)},
    'accuracy_pct': round(test_acc * 100, 4),
    'macro_f1': round(macro_f1, 4),
    'weighted_f1': round(weighted_f1, 4),
    'per_class': {
        cls: {
            'precision': round(report[cls]['precision'], 4),
            'recall':    round(report[cls]['recall'], 4),
            'f1':        round(report[cls]['f1-score'], 4),
            'support':   int(report[cls]['support'])
        } for cls in ['Negative', 'Neutral', 'Positive']
    },
    'train_losses': [round(l, 4) for l in train_losses],
    'val_losses':   [round(l, 4) for l in val_losses],
    'val_accs':     [round(a, 4) for a in val_accs]
}

with open(os.path.join(OUT_DIR, 'distilbert_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT_DIR}/distilbert_results.json")

# ── Confusion matrix figure ───────────────────────────────────────────────────
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
ax.set_title('DistilBERT Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '13_confusion_matrix_distilbert.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix saved.")

# ── Training curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
epochs_range = range(1, EPOCHS + 1)
ax1.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
ax1.plot(epochs_range, val_losses,   'r-o', label='Val Loss')
ax1.set_title('Training and Validation Loss', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
ax2.plot(epochs_range, val_accs, 'g-o', label='Val Accuracy')
ax2.set_title('Validation Accuracy per Epoch', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '14_distilbert_training_curves.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Training curves saved.")

# ── Model comparison figure ───────────────────────────────────────────────────
models = ['Logistic\nRegression', 'Linear\nSVM', 'DistilBERT\n(Multilingual)']
accs   = [92.21, 93.03, round(test_acc * 100, 2)]
f1s    = [0.5596, 0.5625, round(macro_f1, 4)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = ['#4472C4', '#ED7D31', '#70AD47']
bars1 = ax1.bar(models, accs, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_ylim(80, 100)
for bar, val in zip(bars1, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
bars2 = ax2.bar(models, f1s, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_title('Macro F1-Score Comparison', fontsize=13, fontweight='bold')
ax2.set_ylabel('Macro F1-Score', fontsize=11)
ax2.set_ylim(0, 1.0)
for bar, val in zip(bars2, f1s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '15_model_comparison_all.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Model comparison figure saved.")

print("\n✅ DistilBERT training complete!")
print(f"   Accuracy: {test_acc*100:.2f}%")
print(f"   Macro F1: {macro_f1:.4f}")
