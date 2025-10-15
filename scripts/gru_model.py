"""
FINAL: GRU Model with K-Fold Cross-Validation
IMPROVEMENTS:
- Per-fold training plots ✓
- Confusion matrices for validation and test ✓
- Complete visualization suite ✓
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils.preprocessing import preprocess_for_ml_gru
# from utils.evaluation import compute_metrics, print_metrics, plot_confusion_matrix, plot_training_history

# -------------------------------------------------------------
# Dataset Definition
# -------------------------------------------------------------
class HateSpeechDataset(Dataset):
    def __init__(self, input_ids, labels, augment=False):
        self.input_ids = input_ids
        self.labels = labels
        self.augment = augment
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].copy()
        if self.augment and random.random() < 0.15:
            mask = np.random.random(len(input_ids)) > 0.1
            input_ids = [t if m else 0 for t, m in zip(input_ids, mask)]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------------------------------------------------------------
# GRU Classifier
# -------------------------------------------------------------
class OptimizedGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=96, output_dim=4, dropout=0.5):
        super(OptimizedGRUClassifier, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False
        )
        self.embedding_dropout = nn.Dropout(0.3)
        
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding_dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout2(out)
        logits = self.fc2(out)
        return logits

# -------------------------------------------------------------
# Utilities
# -------------------------------------------------------------
def encode_and_pad(tokens, word2idx, max_len=40):
    indices = [word2idx.get(tok, 0) for tok in tokens[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

def train_epoch(model, dataloader, optimizer, device, class_weights):
    model.train()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1

def evaluate(model, dataloader, device, class_weights):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1, all_preds, all_labels

# -------------------------------------------------------------
# Visualization Functions
# -------------------------------------------------------------
def plot_confusion_matrix_custom(y_true, y_pred, labels, save_path, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proportion'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {os.path.basename(save_path)}")


def plot_fold_history(history, fold_num, save_dir):
    """Plot training curves for a single fold"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', 
                linewidth=2, color='#1f77b4', marker='o', markersize=3)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', 
                linewidth=2, color='#ff7f0e', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Fold {fold_num + 1} - Training and Validation Loss', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # F1 plot
    axes[1].plot(epochs, history['train_f1'], label='Train F1', 
                linewidth=2, color='#1f77b4', marker='o', markersize=3)
    axes[1].plot(epochs, history['val_f1'], label='Val F1', 
                linewidth=2, color='#ff7f0e', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title(f'Fold {fold_num + 1} - Training and Validation F1', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'fold_{fold_num + 1}_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Fold {fold_num + 1} training curves saved")


def plot_cv_summary(fold_scores, all_histories, save_dir):
    """Plots K-Fold summary charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Bar chart of fold scores
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, len(fold_scores) + 1), fold_scores, color="#1f77b4", edgecolor='navy')
    plt.axhline(np.mean(fold_scores), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(fold_scores):.4f}")
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, fold_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.01, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel("Fold", fontsize=12)
    plt.ylabel("Validation F1", fontsize=12)
    plt.title("Cross-Validation: Best F1 Score per Fold", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cv_f1_summary.png"), dpi=300)
    plt.close()
    print("✓ CV summary (F1 bar chart) saved")

    # Mean curves across folds
    max_epochs = max(len(h["train_loss"]) for h in all_histories)
    
    def pad_history(hist_list, key):
        padded = []
        for h in hist_list:
            arr = np.array(h[key])
            if len(arr) < max_epochs:
                arr = np.pad(arr, (0, max_epochs - len(arr)), constant_values=np.nan)
            padded.append(arr)
        return np.array(padded)
    
    mean_train_loss = np.nanmean(pad_history(all_histories, "train_loss"), axis=0)
    mean_val_loss = np.nanmean(pad_history(all_histories, "val_loss"), axis=0)
    mean_train_f1 = np.nanmean(pad_history(all_histories, "train_f1"), axis=0)
    mean_val_f1 = np.nanmean(pad_history(all_histories, "val_f1"), axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, max_epochs + 1)
    
    # Average loss
    axes[0].plot(epochs, mean_train_loss, label="Train Loss", linewidth=2, color='#1f77b4')
    axes[0].plot(epochs, mean_val_loss, label="Val Loss", linewidth=2, color='#ff7f0e')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title("Average Train/Val Loss Across Folds", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Average F1
    axes[1].plot(epochs, mean_train_f1, label="Train F1", linewidth=2, color='#1f77b4')
    axes[1].plot(epochs, mean_val_f1, label="Val F1", linewidth=2, color='#ff7f0e')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title("Average Train/Val F1 Across Folds", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cv_training_summary.png"), dpi=300)
    plt.close()
    print("✓ CV training summary (average curves) saved")


def plot_final_training_history(final_history, save_dir):
    """Plot final model train/val loss and F1 curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(final_history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, final_history['train_loss'], label="Train Loss", 
                linewidth=2, color='#1f77b4', marker='o', markersize=4)
    axes[0].plot(epochs, final_history['val_loss'], label="Val Loss", 
                linewidth=2, color='#ff7f0e', marker='s', markersize=4, linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title("Final Model - Training and Validation Loss", fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # F1
    axes[1].plot(epochs, final_history['train_f1'], label="Train F1", 
                linewidth=2, color='#2ca02c', marker='o', markersize=4)
    axes[1].plot(epochs, final_history['val_f1'], label="Val F1", 
                linewidth=2, color='#d62728', marker='s', markersize=4, linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title("Final Model - Training and Validation F1", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gru_final_training_history.png"), dpi=300)
    plt.close()
    print("✓ Final model training curves saved")

# -------------------------------------------------------------
# K-Fold Training Function
# -------------------------------------------------------------
def train_single_fold(train_idx, val_idx, train_df, embedding_matrix, vocab, le, 
                     device, fold_num, save_dir):
    """Train one CV fold"""
    print(f"\n{'='*60}")
    print(f" Fold {fold_num + 1}/5")
    print(f"{'='*60}")
    
    fold_train_df = train_df.iloc[train_idx].copy()
    fold_val_df = train_df.iloc[val_idx].copy()
    
    y_train = le.transform(fold_train_df['Label_Multiclass'])
    y_val = le.transform(fold_val_df['Label_Multiclass'])
    
    train_ds = HateSpeechDataset(fold_train_df['input_ids'].tolist(), y_train, augment=True)
    val_ds = HateSpeechDataset(fold_val_df['input_ids'].tolist(), y_val)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = np.clip(class_weights, 0.5, 4.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = OptimizedGRUClassifier(embedding_matrix, hidden_dim=96, 
                                   output_dim=len(le.classes_), dropout=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    best_val_f1, patience, patience_counter = 0, 5, 0
    hist = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(50):
        tr_loss, tr_f1 = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_loss, val_f1, val_preds, val_labels = evaluate(model, val_loader, device, class_weights)
        scheduler.step()
        
        hist['train_loss'].append(tr_loss)
        hist['val_loss'].append(val_loss)
        hist['train_f1'].append(tr_f1)
        hist['val_f1'].append(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | Gap: {tr_f1-val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_val_preds = val_preds
            best_val_labels = val_labels
            torch.save({'model_state_dict': model.state_dict()}, 
                      os.path.join(save_dir, f'gru_fold_{fold_num}.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    print(f"Fold {fold_num + 1} Best Val F1: {best_val_f1:.4f}")
    
    # Plot fold history
    plot_fold_history(hist, fold_num, save_dir)
    
    # Plot fold confusion matrix
    cm_path = os.path.join(save_dir, f'fold_{fold_num + 1}_confusion_matrix.png')
    plot_confusion_matrix_custom(best_val_labels, best_val_preds, le.classes_, 
                                cm_path, title=f"Fold {fold_num + 1} - Validation Confusion Matrix")
    
    return best_val_f1, hist

# -------------------------------------------------------------
# Main Cross-Validation Training
# -------------------------------------------------------------
def train_with_cross_validation(train_df, val_df, test_df, n_splits=5, save_dir='models/saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(" GRU MODEL WITH 5-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Using device: {device}\n")

    # Ensure tokens exist
    if 'tokens' not in train_df.columns:
        print("ERROR: 'tokens' column missing! Run preprocessing first.")
        return

    # Word2Vec
    print("Building Word2Vec embeddings...")
    all_tokens = train_df['tokens'].tolist()
    w2v = Word2Vec(sentences=all_tokens, vector_size=100, window=5, 
                   min_count=1, workers=4, epochs=10, sg=1)
    vocab = {w: i+1 for i, w in enumerate(w2v.wv.index_to_key)}
    vocab['<PAD>'] = 0
    emb_matrix = np.zeros((len(vocab), 100))
    for w, i in vocab.items():
        if w in w2v.wv:
            emb_matrix[i] = w2v.wv[w]
    
    print(f"Vocabulary size: {len(vocab)}")

    # Encode sequences
    for df in [train_df, val_df, test_df]:
        df['input_ids'] = df['tokens'].apply(lambda x: encode_and_pad(x, vocab, 40))

    le = LabelEncoder()
    le.fit(train_df['Label_Multiclass'])
    print(f"Classes: {le.classes_}\n")

    # ---- K-Fold Cross-Validation ----
    print("="*70)
    print(" STARTING 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores, histories = [], []
    
    for fold, (tr_idx, v_idx) in enumerate(skf.split(train_df, train_df['Label_Multiclass'])):
        best_f1, hist = train_single_fold(tr_idx, v_idx, train_df, emb_matrix, vocab, 
                                          le, device, fold, save_dir)
        fold_scores.append(best_f1)
        histories.append(hist)
    
    # CV Summary
    print("\n" + "="*70)
    print(" CROSS-VALIDATION RESULTS")
    print("="*70)
    for i, score in enumerate(fold_scores):
        print(f"Fold {i+1}: {score:.4f}")
    
    mean_f1, std_f1 = np.mean(fold_scores), np.std(fold_scores)
    print(f"\nMean Val F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*70 + "\n")
    
    plot_cv_summary(fold_scores, histories, save_dir)

    # ---- Final Training with Validation Monitoring ----
    print("="*70)
    print(" TRAINING FINAL MODEL (with validation monitoring)")
    print("="*70 + "\n")
    
    tr_full, val_hold = train_test_split(train_df, test_size=0.1, 
                                         stratify=train_df['Label_Multiclass'], 
                                         random_state=42)
    y_tr = le.transform(tr_full['Label_Multiclass'])
    y_val_hold = le.transform(val_hold['Label_Multiclass'])
    
    tr_loader = DataLoader(HateSpeechDataset(tr_full['input_ids'].tolist(), y_tr, augment=True), 
                          batch_size=64, shuffle=True)
    val_hold_loader = DataLoader(HateSpeechDataset(val_hold['input_ids'].tolist(), y_val_hold), 
                                 batch_size=64)
    
    weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    final_model = OptimizedGRUClassifier(emb_matrix, output_dim=len(le.classes_)).to(device)
    opt = optim.AdamW(final_model.parameters(), lr=1e-4, weight_decay=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=2, factor=0.5)
    
    best_val, patience, counter = 0, 3, 0
    final_hist = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}

    for ep in range(30):
        tr_loss, tr_f1 = train_epoch(final_model, tr_loader, opt, device, weights)
        val_loss, val_f1, val_preds, val_labels = evaluate(final_model, val_hold_loader, device, weights)
        sched.step(val_loss)
        
        final_hist['train_loss'].append(tr_loss)
        final_hist['train_f1'].append(tr_f1)
        final_hist['val_loss'].append(val_loss)
        final_hist['val_f1'].append(val_f1)
        
        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep+1:2d} | Train F1: {tr_f1:.4f} | Val F1: {val_f1:.4f} | Gap: {tr_f1-val_f1:.4f}")
        
        if val_f1 > best_val:
            best_val = val_f1
            counter = 0
            best_val_preds = val_preds
            best_val_labels = val_labels
            torch.save({'model_state_dict': final_model.state_dict(), 'vocab': vocab, 
                       'label_encoder': le}, os.path.join(save_dir, 'gru_final_model.pt'))
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {ep + 1}")
                break

    # Final model plots
    plot_final_training_history(final_hist, save_dir)
    
    # Validation confusion matrix
    val_cm_path = os.path.join(save_dir, 'final_validation_confusion_matrix.png')
    plot_confusion_matrix_custom(best_val_labels, best_val_preds, le.classes_,
                                val_cm_path, title="Final Model - Validation Confusion Matrix")
    
    # ---- Test Set Evaluation ----
    print("\n" + "="*70)
    print(" EVALUATING ON TEST SET")
    print("="*70 + "\n")
    
    y_test = le.transform(test_df['Label_Multiclass'])
    test_loader = DataLoader(HateSpeechDataset(test_df['input_ids'].tolist(), y_test), 
                            batch_size=64)
    
    test_loss, test_f1, test_preds, test_labels = evaluate(final_model, test_loader, device, weights)
    
    # Test confusion matrix
    test_cm_path = os.path.join(save_dir, 'final_test_confusion_matrix.png')
    plot_confusion_matrix_custom(test_labels, test_preds, le.classes_,
                                test_cm_path, title="Final Model - Test Set Confusion Matrix")
    
    # Final Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    print(f"Cross-Validation F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Final Validation F1: {best_val:.4f}")
    print(f"Test Set F1:         {test_f1:.4f}")
    print(f"\nGeneralization: {'✓ Good' if abs(mean_f1 - test_f1) < 0.05 else '⚠ Check variance'}")
    print("="*70 + "\n")
    
    print("✓ All visualizations saved to:", save_dir)
    print("  - fold_X_history.png (5 files)")
    print("  - fold_X_confusion_matrix.png (5 files)")
    print("  - cv_f1_summary.png")
    print("  - cv_training_summary.png")
    print("  - gru_final_training_history.png")
    print("  - final_validation_confusion_matrix.png")
    print("  - final_test_confusion_matrix.png")

# -------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    # Adjust paths as needed
    train_path = "data/processed/train_preprocessed.json"
    val_path = "data/processed/val_preprocessed.json"
    test_path = "data/processed/test_preprocessed.json"

    train_df = pd.read_json(train_path)
    val_df = pd.read_json(val_path)
    test_df = pd.read_json(test_path)

    print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    train_with_cross_validation(train_df, val_df, test_df, n_splits=5)