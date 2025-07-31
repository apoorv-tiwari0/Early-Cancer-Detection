
import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from keras.utils import Sequence, to_categorical
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

#   1. GPU Configuration  
def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPUs configured: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("⚠ No GPUs found, using CPU")
    return len(gpus)

setup_gpu()

#   2. Data Loading  
DATA_DIR = r"C:\\Preprocessed_8_Types"
LABEL_CSV = "labels_multitask_enhanced.csv"

#   Load Data  
df = pd.read_csv(LABEL_CSV)
print(f" Loaded {len(df)} samples")

#   Apply Data Balancing Logic  
def analyze_data_distribution(df):
    print("\n DATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    type_dist = df['cancer_type'].value_counts()
    print("\nCancer Type Distribution:")
    for ct, count in type_dist.items():
        print(f"  {ct}: {count} samples ({count/len(df)*100:.1f}%)")
    
    cancer_dist = df['is_cancerous'].value_counts()
    print(f"\nCancer Detection Distribution:")
    print(f"  Cancerous: {cancer_dist.get(1, 0)} samples ({cancer_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Non-cancerous: {cancer_dist.get(0, 0)} samples ({cancer_dist.get(0, 0)/len(df)*100:.1f}%)")
    
    return type_dist, cancer_dist

def simple_balance_dataset(df):
    print("\n Applying dataset balancing...")
    
    combination_counts = df.groupby(['cancer_type', 'is_cancerous']).size()
    max_count = combination_counts.max()
    
    balanced_dfs = []
    for (cancer_type, is_cancerous), group in df.groupby(['cancer_type', 'is_cancerous']):
        current_count = len(group)
        if current_count < max_count:
            additional_needed = max_count - current_count
            sampled = group.sample(n=additional_needed, replace=True, random_state=42)
            balanced_group = pd.concat([group, sampled], ignore_index=True)
        else:
            balanced_group = group
        balanced_dfs.append(balanced_group)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✓ Dataset balanced from {len(df)} to {len(balanced_df)} samples")
    return balanced_df

# Execute balancing
type_dist, cancer_dist = analyze_data_distribution(df)
df_balanced = simple_balance_dataset(df)

# Create stratification column and train-test split
df_balanced['stratify_col'] = df_balanced['cancer_type'] + '_' + df_balanced['is_cancerous'].astype(str)

train_df, val_df = train_test_split(
    df_balanced, 
    test_size=0.2, 
    stratify=df_balanced['stratify_col'], 
    random_state=42
)

print(f" Training samples: {len(train_df)}")
print(f" Validation samples: {len(val_df)}")

#   Enhanced Preprocessing  
def enhanced_preprocessing(path, img_size=(224,224), augment=True):
    """Enhanced preprocessing with robust augmentations"""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        
        if augment:
            # Geometric augmentations
            if np.random.rand() > 0.3:
                img = cv2.flip(img, np.random.choice([0, 1]))
            
            if np.random.rand() > 0.3:
                angle = np.random.uniform(-30, 30)
                center = (img_size[0]//2, img_size[1]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, img_size)
            
            # Color space augmentations
            if np.random.rand() > 0.3:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hsv[:,:,1] = np.clip(hsv[:,:,1] * np.random.uniform(0.7, 1.3), 0, 255)
                hsv[:,:,2] = np.clip(hsv[:,:,2] * np.random.uniform(0.8, 1.2), 0, 255)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Brightness and contrast
            if np.random.rand() > 0.3:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img * brightness, 0, 255)
            
            if np.random.rand() > 0.3:
                contrast = np.random.uniform(0.8, 1.2)
                img = np.clip(128 + contrast * (img - 128), 0, 255)
            
            # Noise injection
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 5, img.shape)
                img = np.clip(img + noise, 0, 255)
            
            # Zoom and crop
            if np.random.rand() > 0.4:
                zoom_factor = np.random.uniform(0.8, 1.0)
                h, w = img.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                y_start = (h - new_h) // 2
                x_start = (w - new_w) // 2
                img = img[y_start:y_start + new_h, x_start:x_start + new_w]
                img = cv2.resize(img, img_size)
        
        return img.astype(np.float32) / 255.0
    
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

#   Enhanced Data Generator  
class EnhancedMultiTaskGenerator(Sequence):
    def __init__(self, df, batch_size=32, img_size=(224,224), augment=True, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.classes = sorted(df['cancer_type'].unique())
        self.class_map = {label: idx for idx, label in enumerate(self.classes)}
        self.on_epoch_end()
        
        print(f"Generator initialized with {len(self.classes)} cancer types: {self.classes}")
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_imgs, batch_types, batch_cancer = [], [], []
        
        for _, row in batch_df.iterrows():
            img = enhanced_preprocessing(row['path'], self.img_size, self.augment)
            if img is not None:
                batch_imgs.append(img)
                batch_types.append(self.class_map[row['cancer_type']])
                batch_cancer.append(int(row['is_cancerous']))
        
        # Handle empty batches
        if len(batch_imgs) == 0:
            batch_imgs = [np.zeros((*self.img_size, 3), dtype=np.float32)]
            batch_types = [0]
            batch_cancer = [0]
        
        # Convert to numpy arrays with consistent types
        X = np.array(batch_imgs, dtype=np.float32)
        y_types = to_categorical(batch_types, num_classes=len(self.classes)).astype(np.float32)
        y_cancer = np.array(batch_cancer, dtype=np.float32)
        
        return X, {'cancer_type': y_types, 'is_cancerous': y_cancer}

# Create validation generator
val_gen = EnhancedMultiTaskGenerator(val_df, batch_size=16, augment=False, shuffle=False)

#   Rebuild Model Architecture  
def build_enhanced_model(num_classes, stage=3):
    """Rebuild the exact same model architecture"""
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
    base.trainable = True  # Stage 3: Fully unfrozen
    
    x = base.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    
    shared = Dense(512, activation='relu', name='shared_dense_1')(x)
    shared = BatchNormalization(name='shared_bn_1')(shared)
    shared = Dropout(0.4, name='shared_dropout_1')(shared)
    
    shared = Dense(256, activation='relu', name='shared_dense_2')(shared)
    shared = BatchNormalization(name='shared_bn_2')(shared)
    shared = Dropout(0.3, name='shared_dropout_2')(shared)
    
    # Cancer type branch
    type_branch = Dense(128, activation='relu', name='type_dense_1')(shared)
    type_branch = BatchNormalization(name='type_bn_1')(type_branch)
    type_branch = Dropout(0.3, name='type_dropout_1')(type_branch)
    
    type_branch = Dense(64, activation='relu', name='type_dense_2')(type_branch)
    type_branch = BatchNormalization(name='type_bn_2')(type_branch)
    type_branch = Dropout(0.2, name='type_dropout_2')(type_branch)
    
    type_out = Dense(num_classes, activation='softmax', name='cancer_type')(type_branch)
    
    # Cancer detection branch
    cancer_branch = Dense(64, activation='relu', name='cancer_dense_1')(shared)
    cancer_branch = BatchNormalization(name='cancer_bn_1')(cancer_branch)
    cancer_branch = Dropout(0.3, name='cancer_dropout_1')(cancer_branch)
    
    cancer_branch = Dense(32, activation='relu', name='cancer_dense_2')(cancer_branch)
    cancer_branch = BatchNormalization(name='cancer_bn_2')(cancer_branch)
    cancer_branch = Dropout(0.2, name='cancer_dropout_2')(cancer_branch)
    
    cancer_out = Dense(1, activation='sigmoid', name='is_cancerous')(cancer_branch)
    
    return Model(inputs=base.input, outputs=[type_out, cancer_out])

#   Load Pre-trained Model  
print(" Loading pre-trained model...")

model = build_enhanced_model(num_classes=len(val_gen.classes), stage=3)

model.compile(
    optimizer='adam',
    loss={
        'cancer_type': 'categorical_crossentropy', 
        'is_cancerous': 'binary_crossentropy'
    },
    metrics={
        'cancer_type': ['accuracy'],
        'is_cancerous': ['accuracy']
    }
)

try:
    model.load_weights("final_model_weights.h5")
    print(" Successfully loaded final_model_weights.h5")
except Exception as e:
    print(f" Error loading weights: {e}")
    print("Make sure final_model_weights.h5 is in the same directory")
    exit()

#   ENHANCED EVALUATION WITH COMPREHENSIVE VISUALIZATIONS  
def comprehensive_evaluation_with_plots(model, val_generator):
    """Enhanced evaluation function with comprehensive visualizations"""
    print("\n COMPREHENSIVE EVALUATION WITH VISUALIZATIONS")
    print("="*60)
    
    # Process validation data in smaller chunks
    val_imgs, y_true = [], {'cancer_type': [], 'is_cancerous': []}
    
    print(" Collecting validation data...")
    for i in range(len(val_generator)):
        try:
            X, y = val_generator[i]
            val_imgs.append(X)
            y_true['cancer_type'].extend(np.argmax(y['cancer_type'], axis=1))
            y_true['is_cancerous'].extend(y['is_cancerous'])
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(val_generator)} batches")
                
        except Exception as e:
            print(f" Error processing batch {i}: {e}")
            continue
    
    if not val_imgs:
        print(" No validation data could be processed")
        return None
    
    # Make predictions in smaller chunks
    print(" Making predictions...")
    all_predictions = []
    chunk_size = 8
    
    for i in range(0, len(val_imgs), chunk_size):
        try:
            chunk = val_imgs[i:i + chunk_size]
            X_chunk = np.concatenate(chunk) if len(chunk) > 1 else chunk[0]
            
            with tf.device('/CPU:0'):
                pred_chunk = model.predict(X_chunk, batch_size=8, verbose=0)
            
            all_predictions.append(pred_chunk)
            
        except Exception as e:
            print(f" Error predicting chunk {i//chunk_size}: {e}")
            continue
    
    if not all_predictions:
        print(" No predictions could be made")
        return None
    
    # Combine predictions
    try:
        if len(all_predictions[0]) == 2:
            y_pred = [
                np.concatenate([p[0] for p in all_predictions]),
                np.concatenate([p[1] for p in all_predictions])
            ]
        else:
            y_pred = np.concatenate(all_predictions)
    except Exception as e:
        print(f" Error combining predictions: {e}")
        return None
    
    # Calculate metrics
    y_pred_classes = {
        'cancer_type': np.argmax(y_pred[0], axis=1),
        'is_cancerous': (y_pred[1] > 0.5).astype(int).flatten()
    }
    
    # Ensure arrays are same length
    min_length = min(len(y_true['cancer_type']), len(y_pred_classes['cancer_type']))
    y_true['cancer_type'] = y_true['cancer_type'][:min_length]
    y_true['is_cancerous'] = y_true['is_cancerous'][:min_length]
    y_pred_classes['cancer_type'] = y_pred_classes['cancer_type'][:min_length]
    y_pred_classes['is_cancerous'] = y_pred_classes['is_cancerous'][:min_length]
    y_pred_probs = y_pred[1][:min_length]
    
    # Calculate comprehensive metrics
    type_accuracy = accuracy_score(y_true['cancer_type'], y_pred_classes['cancer_type'])
    cancer_accuracy = accuracy_score(y_true['is_cancerous'], y_pred_classes['is_cancerous'])
    
    try:
        f1 = f1_score(y_true['is_cancerous'], y_pred_classes['is_cancerous'], average='weighted')
    except:
        f1 = 0.0
    
    try:
        auc_score = roc_auc_score(y_true['is_cancerous'], y_pred_probs)
    except:
        auc_score = 0.5
    
    #   CREATE COMPREHENSIVE VISUALIZATIONS  
    
    # Create output directory for plots
    os.makedirs('evaluation_plots', exist_ok=True)
    
    # 1. Data Distribution Plots
    create_data_distribution_plots(df_balanced, train_df, val_df)
    
    # 2. Confusion Matrices
    create_confusion_matrices(y_true, y_pred_classes, val_generator.classes)
    
    # 3. ROC and Precision-Recall Curves
    create_roc_pr_curves(y_true['is_cancerous'], y_pred_probs)
    
    # 4. Performance Metrics Visualization
    create_performance_plots(type_accuracy, cancer_accuracy, f1, auc_score)
    
    # 5. Per-Class Performance Analysis
    create_per_class_analysis(y_true, y_pred_classes, val_generator.classes)
    
    # 6. Model Architecture Visualization
    create_model_summary_plot(model)
    
    print("\n📊 All visualization plots saved to 'evaluation_plots/' directory")
    
    return {
        'type_accuracy': type_accuracy,
        'cancer_accuracy': cancer_accuracy,
        'f1_score': f1,
        'auc_roc': auc_score,
        'samples_evaluated': min_length,
        'y_true': y_true,
        'y_pred_classes': y_pred_classes,
        'y_pred_probs': y_pred_probs
    }

#   VISUALIZATION FUNCTIONS  

def create_data_distribution_plots(df_balanced, train_df, val_df):
    """Create comprehensive data distribution plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Original vs Balanced Distribution
    original_dist = df['cancer_type'].value_counts()
    balanced_dist = df_balanced['cancer_type'].value_counts()
    
    axes[0, 0].bar(original_dist.index, original_dist.values, alpha=0.7, color='lightcoral')
    axes[0, 0].set_title('Original Cancer Type Distribution')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(balanced_dist.index, balanced_dist.values, alpha=0.7, color='lightblue')
    axes[0, 1].set_title('Balanced Cancer Type Distribution')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Train vs Validation Split
    train_dist = train_df['cancer_type'].value_counts()
    val_dist = val_df['cancer_type'].value_counts()
    
    x_pos = np.arange(len(train_dist.index))
    width = 0.35
    
    axes[0, 2].bar(x_pos - width/2, train_dist.values, width, label='Train', alpha=0.7, color='green')
    axes[0, 2].bar(x_pos + width/2, val_dist.values, width, label='Validation', alpha=0.7, color='orange')
    axes[0, 2].set_title('Train vs Validation Split')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(train_dist.index, rotation=45)
    axes[0, 2].legend()
    
    # Cancerous vs Non-cancerous Distribution
    cancer_dist = df_balanced['is_cancerous'].value_counts()
    labels = ['Non-cancerous', 'Cancerous']
    colors = ['lightgreen', 'salmon']
    
    axes[1, 0].pie(cancer_dist.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 0].set_title('Cancer vs Non-Cancer Distribution')
    
    # Cancer Type vs Cancer Status Heatmap
    crosstab = pd.crosstab(df_balanced['cancer_type'], df_balanced['is_cancerous'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Cancer Type vs Cancer Status')
    axes[1, 1].set_ylabel('Cancer Type')
    axes[1, 1].set_xlabel('Is Cancerous (0=No, 1=Yes)')
    
    # Sample counts per combination
    combination_counts = df_balanced.groupby(['cancer_type', 'is_cancerous']).size().reset_index(name='count')
    combination_counts['label'] = combination_counts['cancer_type'] + '_' + combination_counts['is_cancerous'].astype(str)
    
    axes[1, 2].bar(range(len(combination_counts)), combination_counts['count'], alpha=0.7, color='purple')
    axes[1, 2].set_title('Samples per Cancer Type-Status Combination')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_xlabel('Combination')
    axes[1, 2].set_xticks(range(len(combination_counts)))
    axes[1, 2].set_xticklabels(combination_counts['label'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrices(y_true, y_pred_classes, class_names):
    """Create confusion matrices for both tasks"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
    
    # Cancer Type Classification Confusion Matrix
    cm_type = confusion_matrix(y_true['cancer_type'], y_pred_classes['cancer_type'])
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Cancer Type Classification')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Cancer Detection Confusion Matrix
    cm_cancer = confusion_matrix(y_true['is_cancerous'], y_pred_classes['is_cancerous'])
    sns.heatmap(cm_cancer, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Non-Cancerous', 'Cancerous'], 
                yticklabels=['Non-Cancerous', 'Cancerous'], ax=axes[1])
    axes[1].set_title('Cancer Detection')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_roc_pr_curves(y_true_cancer, y_pred_probs):
    """Create ROC and Precision-Recall curves for cancer detection"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ROC and Precision-Recall Curves', fontsize=16, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_cancer, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve - Cancer Detection')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true_cancer, y_pred_probs)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve - Cancer Detection')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_plots(type_accuracy, cancer_accuracy, f1, auc_score):
    """Create performance metrics visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # Accuracy Comparison
    accuracies = [type_accuracy, cancer_accuracy]
    labels = ['Cancer Type\nClassification', 'Cancer\nDetection']
    colors = ['skyblue', 'lightcoral']
    
    bars1 = axes[0, 0].bar(labels, accuracies, color=colors, alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Target Achievement
    target = 0.85
    achievement = [type_accuracy/target, cancer_accuracy/target]
    achievement_labels = ['Type Classification\nvs Target (85%)', 'Cancer Detection\nvs Target (85%)']
    
    bars2 = axes[0, 1].bar(achievement_labels, achievement, color=['green' if a >= 1 else 'red' for a in achievement], alpha=0.7)
    axes[0, 1].axhline(y=1, color='black', linestyle='--', label='Target (100%)')
    axes[0, 1].set_title('Target Achievement')
    axes[0, 1].set_ylabel('Achievement Ratio')
    axes[0, 1].legend()
    
    # Add value labels
    for bar, ach in zip(bars2, achievement):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{ach:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # Overall Performance Metrics
    metrics = [type_accuracy, cancer_accuracy, f1, auc_score]
    metric_labels = ['Type\nAccuracy', 'Cancer\nAccuracy', 'F1\nScore', 'AUC-ROC']
    
    bars3 = axes[1, 0].bar(metric_labels, metrics, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Overall Performance Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0, 1])
    
    # Add value labels
    for bar, metric in zip(bars3, metrics):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{metric:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Performance Radar Chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    metrics_radar = metrics + [metrics[0]]  # Complete the circle
    angles += angles[:1]
    
    axes[1, 1] = plt.subplot(2, 2, 4, projection='polar')
    axes[1, 1].plot(angles, metrics_radar, 'o-', linewidth=2, color='blue', alpha=0.7)
    axes[1, 1].fill(angles, metrics_radar, color='blue', alpha=0.25)
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(metric_labels)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Performance Radar Chart', pad=20)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_per_class_analysis(y_true, y_pred_classes, class_names):
    """Create per-class performance analysis"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate per-class metrics
    precisions = precision_score(y_true['cancer_type'], y_pred_classes['cancer_type'], average=None, zero_division=0)
    recalls = recall_score(y_true['cancer_type'], y_pred_classes['cancer_type'], average=None, zero_division=0)
    f1_scores = f1_score(y_true['cancer_type'], y_pred_classes['cancer_type'], average=None, zero_division=0)
    
    # Create DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Cancer Type': class_names,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Per-Class Performance Analysis', fontsize=16, fontweight='bold')
    
    # Bar plot of all metrics
    x_pos = np.arange(len(class_names))
    width = 0.25
    
    axes[0, 0].bar(x_pos - width, precisions, width, label='Precision', alpha=0.8, color='blue')
    axes[0, 0].bar(x_pos, recalls, width, label='Recall', alpha=0.8, color='green')
    axes[0, 0].bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8, color='red')
    
    axes[0, 0].set_xlabel('Cancer Type')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Per-Class Metrics Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Heatmap of confusion matrix with percentages
    cm = confusion_matrix(y_true['cancer_type'], y_pred_classes['cancer_type'])
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix (Percentage)')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # Individual metric plots
    axes[1, 0].barh(class_names, precisions, color='blue', alpha=0.7)
    axes[1, 0].set_title('Precision by Cancer Type')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_xlim([0, 1])
    
    axes[1, 1].barh(class_names, f1_scores, color='red', alpha=0.7)
    axes[1, 1].set_title('F1-Score by Cancer Type')
    axes[1, 1].set_xlabel('F1-Score')
    axes[1, 1].set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/per_class_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed per-class report
    print("\n📊 DETAILED PER-CLASS PERFORMANCE")
    print("="*60)
    print(f"{'Cancer Type':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*60)
    for i, cancer_type in enumerate(class_names):
        print(f"{cancer_type:<15} {precisions[i]:<10.3f} {recalls[i]:<10.3f} {f1_scores[i]:<10.3f}")

def create_model_summary_plot(model):
    """Create model architecture summary visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('Model Architecture Summary', fontsize=16, fontweight='bold')
    
    # Model layer information
    layer_names = []
    layer_params = []
    layer_types = []
    
    for layer in model.layers:
        if hasattr(layer, 'count_params'):
            layer_names.append(layer.name[:15] + '...' if len(layer.name) > 15 else layer.name)
            layer_params.append(layer.count_params())
            layer_types.append(type(layer).__name__)
    
    # Parameters by layer type
    layer_type_params = {}
    for ltype, params in zip(layer_types, layer_params):
        if ltype in layer_type_params:
            layer_type_params[ltype] += params
        else:
            layer_type_params[ltype] = params
    
    # Plot parameters by layer type
    axes[0].pie(layer_type_params.values(), labels=layer_type_params.keys(), autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Parameters Distribution by Layer Type')
    
    # Model complexity metrics
    total_params = model.count_params()
    trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
    non_trainable_params = total_params - trainable_params
    
    complexity_data = [trainable_params, non_trainable_params]
    complexity_labels = ['Trainable', 'Non-trainable']
    colors = ['lightblue', 'lightcoral']
    
    bars = axes[1].bar(complexity_labels, complexity_data, color=colors, alpha=0.7)
    axes[1].set_title('Model Parameters Breakdown')
    axes[1].set_ylabel('Number of Parameters')
    
    # Add value labels
    for bar, value in zip(bars, complexity_data):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(complexity_data)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/model_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print model summary
    print("\n MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print(f"Model Depth: {len(model.layers)} layers")

#   RUN ENHANCED EVALUATION  
print("\n🚀 EVALUATING PRE-TRAINED MODEL WITH COMPREHENSIVE VISUALIZATIONS")
print("="*70)

evaluation_results = comprehensive_evaluation_with_plots(model, val_gen)

if evaluation_results:
    #   Final Results Display  
    print("\n" + "="*60)
    print(" FINAL EVALUATION RESULTS")
    print("="*60)
    print(f" Samples Evaluated: {evaluation_results['samples_evaluated']}")
    print(f" Cancer Type Classification Accuracy: {evaluation_results['type_accuracy']:.4f}")
    print(f" Cancer Detection Accuracy: {evaluation_results['cancer_accuracy']:.4f}")
    print(f" Weighted F1 Score: {evaluation_results['f1_score']:.4f}")
    print(f" AUC-ROC Score: {evaluation_results['auc_roc']:.4f}")
    
    # Performance assessment
    target_achieved = evaluation_results['type_accuracy'] >= 0.85
    print(f"\n Target Accuracy (≥0.85): {' ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    if target_achieved:
        print(" Your pre-trained model has achieved the target performance!")
    else:
        improvement_needed = 0.85 - evaluation_results['type_accuracy']
        print(f" Current model needs {improvement_needed:.1%} improvement to reach target")
    
    print("\n All visualization plots saved to 'evaluation_plots/' directory")
    print("="*60)
else:
    print(" Evaluation failed. Check your data and model files.")

print("\n Comprehensive evaluation with visualizations completed!")
