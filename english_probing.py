"""
Chain-of-Thought Success Prediction using Probing Classifiers
Replicating the methodology from "Knowing Before Saying" paper
MODIFIED: Multiclass classification instead of binary with threshold
FIXED: Multi-GPU support and memory optimization
ADDED: CoT prompt customization and capability filtering
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')


# ==================== Probing Classifier ====================
class ProbeClassifier(nn.Module):
    """
    Feedforward neural network probe for multiclass classification.
    Architecture: 3 hidden layers (256, 128, 64) with ReLU, output with Softmax
    """
    def __init__(self, input_dim: int = 4096, num_classes: int = 5, dropout: float = 0.1):
        super(ProbeClassifier, self).__init__()
        self.num_classes = num_classes
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


# ==================== Dataset ====================
class HiddenStateDataset(Dataset):
    """Dataset for hidden states and labels"""
    def __init__(self, hidden_states: np.ndarray, labels: np.ndarray):
        self.hidden_states = torch.FloatTensor(hidden_states)
        self.labels = torch.LongTensor(labels)  # LongTensor for multiclass
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.hidden_states[idx], self.labels[idx]


# ==================== Hidden State Extractor ====================
class HiddenStateExtractor:
    """Extract hidden states from LLM at different generation stages"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with memory optimization
        print(f"Loading model on device: {device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Auto distribute across GPUs
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            offload_folder="/mnt/storage/deeksha/offload",
            max_memory={i: "10GiB" for i in range(torch.cuda.device_count())}  # Limit per GPU
        )
        self.model.eval()
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Find which device the model's first layer is on
        self.model_device = next(self.model.parameters()).device
        print(f"Model loaded on device(s): {self.model_device}")
    
    def extract_hidden_states(
        self, 
        text: str, 
        layer_idx: int = -1,
        position: str = "last"
    ) -> np.ndarray:
        """
        Extract hidden states from a specific layer
        
        Args:
            text: Input text
            layer_idx: Which layer to extract (-1 for last layer)
            position: Which token position ("last", "mean")
        
        Returns:
            Hidden state vector
        """
        # Tokenize on CPU first
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=False
        )
        
        # Move to the same device as model's first parameter
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            
            if position == "last":
                # Use last token's representation
                hidden_state = hidden_states[0, -1, :].cpu().float().numpy()
            elif position == "mean":
                # Average across sequence
                hidden_state = hidden_states[0].mean(dim=0).cpu().float().numpy()
            else:
                raise ValueError(f"Unknown position: {position}")
            
            # Clean up GPU memory
            del outputs, hidden_states
            torch.cuda.empty_cache()
        
        return hidden_state
    
    def extract_partial_generation(
        self,
        prompt: str,
        full_response: str,
        percentage: int,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        Extract hidden states at percentage% of generation
        
        Args:
            prompt: Initial prompt
            full_response: Complete response
            percentage: How much of response to include (0-100)
            layer_idx: Which layer to extract
        
        Returns:
            Hidden state vector
        """
        if percentage == 0:
            text = prompt
        else:
            # Tokenize to split at token level
            response_tokens = self.tokenizer.encode(
                full_response, 
                add_special_tokens=False
            )
            cutoff = int(len(response_tokens) * percentage / 100)
            partial_response = self.tokenizer.decode(
                response_tokens[:cutoff], 
                skip_special_tokens=True
            )
            text = prompt + partial_response
        
        return self.extract_hidden_states(text, layer_idx)


# ==================== Training Functions ====================
def train_probe(
    model: ProbeClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    optimizer_name: str,
    weight_init: str,
    device: str,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[ProbeClassifier, Dict]:
    """Train the probing classifier"""
    
    # Initialize weights
    if weight_init == "HE_uniform":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif weight_init == "HE_normal":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
    model = model.to(device)
    
    # Use CrossEntropyLoss for multiclass classification
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses, train_preds, train_labels = [], [], []
        
        for hidden_states, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            hidden_states = hidden_states.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(hidden_states)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        
        with torch.no_grad():
            for hidden_states, labels in val_loader:
                hidden_states = hidden_states.to(device)
                labels = labels.to(device)
                
                outputs = model(hidden_states)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        
        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(np.mean(val_losses))
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model, history


def evaluate_probe(
    model: ProbeClassifier,
    test_loader: DataLoader,
    device: str,
    class_names: Optional[List[str]] = None
) -> Dict:
    """Evaluate the probe on test set"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for hidden_states, labels in test_loader:
            hidden_states = hidden_states.to(device)
            outputs = model(hidden_states)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(model.num_classes)]
    
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


# ==================== Data Processing ====================
def prepare_prompt(instruction: str, system_prompt: str = "") -> str:
    """Add CoT prompt to instruction"""
    return f"{system_prompt}\n\n{instruction}"


def filter_by_capability(df: pd.DataFrame, capability_column: str, capability_value: str) -> pd.DataFrame:
    """
    Filter dataframe by capability
    
    Args:
        df: Input dataframe
        capability_column: Name of the column containing capabilities
        capability_value: Specific capability to filter for
    
    Returns:
        Filtered dataframe
    """
    if capability_column not in df.columns:
        print(f"Warning: Column '{capability_column}' not found in dataframe. Using all data.")
        return df
    
    filtered_df = df[df[capability_column] == capability_value].copy()
    return filtered_df


def create_multiclass_label(
    score: float,
    num_classes: int = 5,
    score_range: Tuple[float, float] = (1.0, 5.0)
) -> int:
    """
    Create multiclass label from continuous score
    
    Args:
        score: The score value
        num_classes: Number of classes (default: 5 for scores 1-5)
        score_range: Range of scores (min, max)
    
    Returns:
        Class label (0-indexed)
    """
    min_score, max_score = score_range
    
    # Clamp score to valid range
    score = max(min_score, min(max_score, score))
    
    # Convert to 0-indexed class
    # For scores 1-5 with 5 classes: score 1 -> class 0, score 5 -> class 4
    normalized = (score - min_score) / (max_score - min_score)
    class_idx = int(normalized * (num_classes - 0.0001))  # Avoid class == num_classes
    class_idx = min(class_idx, num_classes - 1)  # Ensure within bounds
    
    return class_idx


def process_dataframe(
    df: pd.DataFrame,
    extractor: HiddenStateExtractor,
    layer_idx: int,
    percentage: int = 0,
    max_samples: Optional[int] = None,
    num_classes: int = 5,
    score_range: Tuple[float, float] = (1.0, 5.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process dataframe to extract hidden states and labels
    
    Args:
        df: Input dataframe with your columns
        extractor: Hidden state extractor
        layer_idx: Which layer to extract from
        percentage: Generation percentage (0 for before generation)
        max_samples: Limit number of samples
        num_classes: Number of classes for classification
        score_range: Range of scores in the data
    
    Returns:
        hidden_states, labels
    """
    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    hidden_states = []
    labels = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting hidden states"):
        try:
            # Prepare prompt
            instruction = row['input']
            response = row['orig_response']
            score = row['orig_score']
            system_prompt = row['system_prompt']
            
            prompt = prepare_prompt(instruction, system_prompt)
            
            # Extract hidden states
            if percentage == 0:
                # Before generation - just prompt
                hidden_state = extractor.extract_hidden_states(prompt, layer_idx)
            else:
                # During generation
                hidden_state = extractor.extract_partial_generation(
                    prompt, response, percentage, layer_idx
                )
            
            # Create multiclass label
            label = create_multiclass_label(score, num_classes, score_range)
            
            hidden_states.append(hidden_state)
            labels.append(label)
            
            # Clear cache every 10 samples
            if idx % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return np.array(hidden_states), np.array(labels)


def balance_dataset(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance dataset to have equal samples per class"""
    unique_labels = np.unique(labels)
    class_counts = [np.sum(labels == label) for label in unique_labels]
    min_count = min(class_counts)
    
    balanced_indices = []
    np.random.seed(random_state)
    
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        sampled_indices = np.random.choice(label_indices, min_count, replace=False)
        balanced_indices.extend(sampled_indices)
    
    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    return hidden_states[balanced_indices], labels[balanced_indices]


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute class weights for imbalanced datasets"""
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    class_weights = total / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(class_weights)


# ==================== Main Experiment Runner ====================
def run_experiment(args):
    """Run complete probing experiment"""
    
    print(f"\n{'='*60}")
    print(f"Running CoT Success Prediction Experiment (Multiclass)")
    print(f"Model: {args.model_name}")
    print(f"Layer: {args.layer_idx}")
    print(f"Generation %: {args.percentage}")
    print(f"Number of classes: {args.num_classes}")
    if args.filter_capability:
        print(f"Capability Filter: {args.capability_column}={args.filter_capability}")
    print(f"{'='*60}\n")
    
    # Setup - use cpu for probe training to avoid device conflicts
    probe_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    # Filter by capability if requested
    if args.filter_capability:
        df = filter_by_capability(df, args.capability_column, args.filter_capability)
        if len(df) == 0:
            print(f"Error: No samples found for {args.capability_column}={args.filter_capability}")
            return None, None, None
    
    # Initialize extractor (will use auto device_map for multi-GPU)
    print(f"\nLoading model: {args.model_name}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    extractor = HiddenStateExtractor(args.model_name)
    
    # Extract hidden states
    print(f"\nExtracting hidden states from layer {args.layer_idx}...")
    hidden_states, labels = process_dataframe(
        df, 
        extractor, 
        args.layer_idx, 
        args.percentage,
        args.max_samples,
        args.num_classes,
        args.score_range,
    )
    
    print(f"Extracted {len(hidden_states)} samples")
    for class_idx in range(args.num_classes):
        count = np.sum(labels == class_idx)
        print(f"Class {class_idx}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Free up memory after extraction
    del extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Balance dataset if requested
    if args.balance_data:
        print("\nBalancing dataset...")
        hidden_states, labels = balance_dataset(hidden_states, labels)
        print(f"Balanced to {len(hidden_states)} samples")
        for class_idx in range(args.num_classes):
            count = np.sum(labels == class_idx)
            print(f"Class {class_idx}: {count} samples")
    
    # Compute class weights if using weighted loss
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(labels, args.num_classes)
        print(f"\nClass weights: {class_weights}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        hidden_states, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = HiddenStateDataset(X_train, y_train)
    val_dataset = HiddenStateDataset(X_val, y_val)
    test_dataset = HiddenStateDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    
    # Initialize and train probe
    print("\nInitializing probe...")
    input_dim = hidden_states.shape[1]
    probe = ProbeClassifier(input_dim=input_dim, num_classes=args.num_classes, dropout=args.dropout)
    
    print(f"\nTraining probe for {args.num_epochs} epochs on {probe_device}...")
    probe, history = train_probe(
        probe,
        train_loader,
        val_loader,
        args.num_epochs,
        args.learning_rate,
        args.optimizer,
        args.weight_init,
        probe_device,
        class_weights
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    class_names = [f"Score_{i+1}" for i in range(args.num_classes)]
    results = evaluate_probe(probe, test_loader, probe_device, class_names)
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['precision_macro']:.4f}")
    print(f"Macro Recall: {results['recall_macro']:.4f}")
    print(f"Macro F1 Score: {results['f1_macro']:.4f}")
    print(f"\nPer-Class Metrics:")
    for i in range(args.num_classes):
        print(f"  Class {i} (Score {i+1}):")
        print(f"    Precision: {results['precision_per_class'][i]:.4f}")
        print(f"    Recall: {results['recall_per_class'][i]:.4f}")
        print(f"    F1: {results['f1_per_class'][i]:.4f}")
        print(f"    Support: {results['support_per_class'][i]}")
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print(f"{'='*60}\n")
    
    # Create filename suffix based on capability filter
    capability_suffix = f"_{args.filter_capability}" if args.filter_capability else ""
    
    # Save results
    save_path = output_dir / f"probe_layer{args.layer_idx}_pct{args.percentage}{capability_suffix}.pt"
    torch.save({
        'model_state_dict': probe.state_dict(),
        'args': vars(args),
        'history': history,
        'test_results': {k: v for k, v in results.items() 
                        if k not in ['predictions', 'labels', 'probabilities']},
        'input_dim': input_dim,
        'num_classes': args.num_classes
    }, save_path)
    
    print(f"Saved probe to {save_path}")
    
    # Save detailed results
    results_path = output_dir / f"results_layer{args.layer_idx}_pct{args.percentage}{capability_suffix}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_to_save = {
        'args': vars(args),
        'test_accuracy': float(results['accuracy']),
        'test_precision_macro': float(results['precision_macro']),
        'test_recall_macro': float(results['recall_macro']),
        'test_f1_macro': float(results['f1_macro']),
        'per_class_precision': results['precision_per_class'].tolist(),
        'per_class_recall': results['recall_per_class'].tolist(),
        'per_class_f1': results['f1_per_class'].tolist(),
        'per_class_support': results['support_per_class'].tolist(),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'train_history': history
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"Saved results to {results_path}")
    
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    
    return probe, results, history


# ==================== Multi-Layer Experiment ====================
def run_all_layers_experiment(args):
    """Run experiment across all layers to find best performing ones"""
    
    # Load data once
    print("Loading dataset...")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    # Filter by capability if requested
    if args.filter_capability:
        df = filter_by_capability(df, args.capability_column, args.filter_capability)
        if len(df) == 0:
            print(f"Error: No samples found for {args.capability_column}={args.filter_capability}")
            return
    
    # Initialize extractor to get layer count
    print(f"Loading model: {args.model_name}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    extractor = HiddenStateExtractor(args.model_name)
    
    # Determine number of layers
    num_layers = extractor.model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")
    
    # Clean up
    del extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    results_all_layers = {}
    
    for layer_idx in range(num_layers):
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx}/{num_layers-1}")
        print(f"{'='*60}")
        
        # Update args for this layer
        args.layer_idx = layer_idx
        
        try:
            _, results, _ = run_experiment(args)
            if results is not None:
                results_all_layers[layer_idx] = results['accuracy']
            else:
                results_all_layers[layer_idx] = 0.0
            
        except Exception as e:
            print(f"Error processing layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            results_all_layers[layer_idx] = 0.0
        
        # Clean up between layers
        torch.cuda.empty_cache()
        gc.collect()
    
    # Find best layers
    sorted_layers = sorted(results_all_layers.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print("Results across all layers:")
    print(f"{'='*60}")
    for layer, acc in sorted_layers[:10]:
        print(f"Layer {layer}: {acc:.4f}")
    
    # Create filename suffix based on capability filter
    capability_suffix = f"_{args.filter_capability}" if args.filter_capability else ""
    
    # Save summary
    summary_path = Path(args.output_dir) / f"all_layers_summary{capability_suffix}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'results_by_layer': results_all_layers,
            'top_5_layers': [layer for layer, _ in sorted_layers[:5]],
            'model_name': args.model_name,
            'num_classes': args.num_classes,
            'capability_filter': args.filter_capability,
        }, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")


# ==================== Command Line Interface ====================
def main():
    parser = argparse.ArgumentParser(
        description="CoT Success Prediction using Probing Classifiers (Multiclass)"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to CSV file with your dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Directory to save results"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM to use"
    )
    
    # Experiment settings
    parser.add_argument(
        "--layer_idx", type=int, default=14,
        help="Layer index to extract hidden states from (-1 for last layer)"
    )
    parser.add_argument(
        "--percentage", type=int, default=0,
        help="Percentage of generation (0 for before generation, 10-100 for during)"
    )
    parser.add_argument(
        "--all_layers", action="store_true",
        help="Run experiment on all layers"
    )
    
    # Multiclass settings
    parser.add_argument(
        "--num_classes", type=int, default=5,
        help="Number of classes for classification (default: 5 for scores 1-5)"
    )
    parser.add_argument(
        "--score_range", type=float, nargs=2, default=[1.0, 5.0],
        help="Range of scores in the data (min max)"
    )
    parser.add_argument(
        "--use_class_weights", action="store_true",
        help="Use class weights to handle imbalanced data"
    )
    
    # Capability filtering
    parser.add_argument(
        "--capability_column", type=str, default="capability",
        help="Name of the column containing capability information"
    )
    parser.add_argument(
        "--filter_capability", type=str, default=None,
        choices=["grounding", "instruction_following", "planning", 
                 "reasoning", "safety", "theory_of_mind", None],
        help="Filter data by specific capability (e.g., 'reasoning', 'planning')"
    )
    
    # Data processing
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--balance_data", action="store_true",
        help="Balance samples across all classes"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use"
    )
    parser.add_argument(
        "--weight_init", type=str, default="HE_uniform",
        choices=["HE_uniform", "HE_normal", "default"],
        help="Weight initialization strategy"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate"
    )
    
    args = parser.parse_args()
    
    # Convert score_range to tuple
    args.score_range = tuple(args.score_range)
    
    # Run experiment
    if args.all_layers:
        run_all_layers_experiment(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()