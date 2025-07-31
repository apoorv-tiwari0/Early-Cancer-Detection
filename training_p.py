
import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import Sequence, to_categorical
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

#  GPU Configuration  
def setup_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f" GPUs configured: {len(gpus)}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print(" No GPUs found, using CPU")
    return len(gpus)

setup_gpu()

#  Enhanced Data Preparation  
DATA_DIR = r"C:\\Preprocessed_8_Types"
LABEL_CSV = "labels_multitask_enhanced.csv"

def validate_and_clean_data(df):
    """Validate data integrity and remove corrupted files"""
    print(" Validating data files...")
    valid_rows = []
    corrupted_count = 0
    
    for idx, row in df.iterrows():
        if os.path.exists(row['path']):
            try:
                img = cv2.imread(row['path'])
                if img is not None and img.shape[0] > 50 and img.shape[1] > 50:
                    valid_rows.append(row)
                else:
                    corrupted_count += 1
            except Exception:
                corrupted_count += 1
        else:
            corrupted_count += 1
    
    print(f" Valid files: {len(valid_rows)}")
    print(f" Corrupted/missing files: {corrupted_count}")
    return pd.DataFrame(valid_rows).reset_index(drop=True)

def create_enhanced_labels():
    """Create enhanced labels with validation"""
    if not os.path.exists(LABEL_CSV):
        print(" Creating enhanced labels...")
        rows = []
        
        for ct in os.listdir(DATA_DIR):
            folder = os.path.join(DATA_DIR, ct)
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rows.append({
                            'path': os.path.join(folder, f),
                            'cancer_type': ct,
                            'is_cancerous': int(f.startswith('cancerous')),
                            'file_size': os.path.getsize(os.path.join(folder, f))
                        })
        
        df = pd.DataFrame(rows)
        df = validate_and_clean_data(df)
        df.to_csv(LABEL_CSV, index=False)
        print(f" Saved {len(df)} validated samples to {LABEL_CSV}")
    else:
        df = pd.read_csv(LABEL_CSV)
        print(f" Loaded {len(df)} samples from existing {LABEL_CSV}")
    
    return df

df = create_enhanced_labels()

#    Data Analysis and Balancing  
def analyze_data_distribution(df):
    """Analyze and display data distribution"""
    print("\n DATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Cancer type distribution
    type_dist = df['cancer_type'].value_counts()
    print("\nCancer Type Distribution:")
    for ct, count in type_dist.items():
        print(f"  {ct}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Cancer vs Non-cancer distribution
    cancer_dist = df['is_cancerous'].value_counts()
    print(f"\nCancer Detection Distribution:")
    print(f"  Cancerous: {cancer_dist.get(1, 0)} samples ({cancer_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Non-cancerous: {cancer_dist.get(0, 0)} samples ({cancer_dist.get(0, 0)/len(df)*100:.1f}%)")
    
    return type_dist, cancer_dist

type_dist, cancer_dist = analyze_data_distribution(df)

def balance_dataset(df):
    """Balance dataset using oversampling - FIXED VERSION"""
    print("\n Balancing dataset...")
    
    # **FIX: Properly create combined target for stratification**
    # Convert is_cancerous to string to avoid confusion
    df_copy = df.copy()
    df_copy['cancer_status'] = df_copy['is_cancerous'].astype(str)  # '0' or '1'
    df_copy['combined_target'] = df_copy['cancer_type'] + '_' + df_copy['cancer_status']
    
    print(" Sample combined targets:")
    print(df_copy['combined_target'].value_counts().head())
    
    # Apply oversampling on the features we want to keep
    features_to_balance = ['path', 'cancer_type', 'is_cancerous']
    if 'file_size' in df_copy.columns:
        features_to_balance.append('file_size')
    
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(
        df_copy[features_to_balance], 
        df_copy['combined_target']
    )
    
    # **FIX: Reconstruct balanced dataframe properly**
    balanced_df = X_balanced.copy()  # This already contains all the original columns
    
    print(f"✓ Dataset balanced from {len(df)} to {len(balanced_df)} samples")
    
    # Verify the results
    print("\n Balanced distribution:")
    print("Cancer Type Distribution:")
    for ct, count in balanced_df['cancer_type'].value_counts().items():
        print(f"  {ct}: {count} samples")
    
    print("\nCancer Status Distribution:")
    cancer_balanced = balanced_df['is_cancerous'].value_counts()
    print(f"  Non-cancerous (0): {cancer_balanced.get(0, 0)} samples")
    print(f"  Cancerous (1): {cancer_balanced.get(1, 0)} samples")
    
    return balanced_df

# Alternative simpler balancing function if the above still causes issues
def simple_balance_dataset(df):
    """Simple balancing using pandas groupby"""
    print("\n Applying simple dataset balancing...")
    
    # Find the maximum count for any cancer_type + is_cancerous combination
    combination_counts = df.groupby(['cancer_type', 'is_cancerous']).size()
    max_count = combination_counts.max()
    
    print(f" Target count per combination: {max_count}")
    
    balanced_dfs = []
    for (cancer_type, is_cancerous), group in df.groupby(['cancer_type', 'is_cancerous']):
        current_count = len(group)
        if current_count < max_count:
            # Oversample this group
            additional_needed = max_count - current_count
            sampled = group.sample(n=additional_needed, replace=True, random_state=42)
            balanced_group = pd.concat([group, sampled], ignore_index=True)
        else:
            balanced_group = group
        
        balanced_dfs.append(balanced_group)
        print(f"  {cancer_type}_{is_cancerous}: {current_count} → {len(balanced_group)} samples")
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ Dataset balanced from {len(df)} to {len(balanced_df)} samples")
    return balanced_df

# Try the fixed version first, fall back to simple version if it fails
try:
    df_balanced = balance_dataset(df)
except Exception as e:
    print(f" Advanced balancing failed: {e}")
    print(" Falling back to simple balancing method...")
    df_balanced = simple_balance_dataset(df)

#    Enhanced Preprocessing  
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

#    Enhanced Data Generator  
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

#   Train-Test Split with Stratification  
print(" Creating train-validation split...")

# Create stratification column for train-test split
df_balanced['stratify_col'] = df_balanced['cancer_type'] + '_' + df_balanced['is_cancerous'].astype(str)

train_df, val_df = train_test_split(
    df_balanced, 
    test_size=0.2, 
    stratify=df_balanced['stratify_col'], 
    random_state=42
)

print(f" Training samples: {len(train_df)}")
print(f" Validation samples: {len(val_df)}")

train_gen = EnhancedMultiTaskGenerator(train_df, batch_size=16, augment=True)
val_gen = EnhancedMultiTaskGenerator(val_df, batch_size=16, augment=False, shuffle=False)

#   Enhanced Model Architecture  
def build_enhanced_model(num_classes, stage=1):
    """Build enhanced model with progressive training capability"""
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224,224,3))
    
    # Progressive unfreezing strategy
    if stage == 1:
        # Stage 1: Freeze most layers, unfreeze last 20
        for layer in base.layers[:-20]:
            layer.trainable = False
        print(" Stage 1: Frozen base layers except last 20")
    elif stage == 2:
        # Stage 2: Unfreeze last 50 layers
        for layer in base.layers[:-50]:
            layer.trainable = False
        for layer in base.layers[-50:]:
            layer.trainable = True
        print(" Stage 2: Unfrozen last 50 layers")
    else:
        # Stage 3: Full unfreezing
        base.trainable = True
        print(" Stage 3: Fully unfrozen base model")
    
    # Enhanced architecture
    x = base.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='batch_norm_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)
    
    # Shared feature extraction with more capacity
    shared = Dense(512, activation='relu', name='shared_dense_1')(x)
    shared = BatchNormalization(name='shared_bn_1')(shared)
    shared = Dropout(0.4, name='shared_dropout_1')(shared)
    
    shared = Dense(256, activation='relu', name='shared_dense_2')(shared)
    shared = BatchNormalization(name='shared_bn_2')(shared)
    shared = Dropout(0.3, name='shared_dropout_2')(shared)
    
    # Cancer type classification branch
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
    
    model = Model(inputs=base.input, outputs=[type_out, cancer_out], name='enhanced_cancer_classifier')
    return model

#    FIXED Class Weight Calculation  
def calculate_class_weights(df):
    """Calculate class weights for balanced training - FIXED VERSION"""
    print(" Calculating class weights...")
    
    try:
        # **FIX: Convert classes to numpy array**
        cancer_classes = np.array(sorted(df['cancer_type'].unique()))
        print(f" Cancer classes: {cancer_classes}")
        
        type_weights = compute_class_weight(
            'balanced', 
            classes=cancer_classes,  # Now it's a numpy array
            y=df['cancer_type'].values
        )
        
        # Create class weight dictionary mapping class indices to weights
        type_weight_dict = {}
        for i, class_name in enumerate(cancer_classes):
            type_weight_dict[i] = type_weights[i]
        
        print(" Cancer type weights:")
        for i, (class_name, weight) in enumerate(zip(cancer_classes, type_weights)):
            print(f"  {class_name} (index {i}): {weight:.3f}")
        
        # **FIX: Cancer detection weights**
        cancer_detection_classes = np.array([0, 1])
        cancer_weights = compute_class_weight(
            'balanced',
            classes=cancer_detection_classes,
            y=df['is_cancerous'].values
        )
        
        cancer_weight_dict = {0: cancer_weights[0], 1: cancer_weights[1]}
        
        print(f" Cancer detection weights:")
        print(f"  Non-cancerous (0): {cancer_weights[0]:.3f}")
        print(f"  Cancerous (1): {cancer_weights[1]:.3f}")
        
        print(" Class weights calculated successfully")
        return type_weight_dict, cancer_weight_dict
        
    except Exception as e:
        print(f" Error calculating class weights: {e}")
        print(" Using uniform weights as fallback")
        
        # Fallback to uniform weights
        num_classes = len(df['cancer_type'].unique())
        type_weight_dict = {i: 1.0 for i in range(num_classes)}
        cancer_weight_dict = {0: 1.0, 1: 1.0}
        
        return type_weight_dict, cancer_weight_dict

type_weights, cancer_weights = calculate_class_weights(train_df)

#    Custom Callbacks  
class DetailedProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.stage_name = "Training"
    
    def set_stage(self, stage_name):
        self.stage_name = stage_name
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\n{self.stage_name} - Epoch {epoch+1} Summary:")
        print(f"  Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        print(f"  Type Acc: {logs.get('cancer_type_accuracy', 0):.4f} | Val Type Acc: {logs.get('val_cancer_type_accuracy', 0):.4f}")
        print(f"  Cancer Acc: {logs.get('is_cancerous_accuracy', 0):.4f} | Val Cancer Acc: {logs.get('val_is_cancerous_accuracy', 0):.4f}")

def cosine_decay_with_warmup(epoch, lr):
    """Custom learning rate schedule with warmup"""
    warmup_epochs = 3
    total_epochs = 20
    
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr * 0.5 * (1 + np.cos(np.pi * progress))

class SafeModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is not None:
            if self.save_best_only:
                if current < self.best:
                    self.best = current
                    try:
                        self.model.save_weights(self.filepath)
                        if self.verbose:
                            print(f"\n💾 Epoch {epoch+1}: {self.monitor} improved to {current:.5f}, saving weights")
                    except Exception as e:
                        print(f" Error saving weights: {e}")
            else:
                try:
                    self.model.save_weights(self.filepath)
                except Exception as e:
                    print(f" Error saving weights: {e}")

#    Multi-Stage Training Function  
def multi_stage_training():
    """Execute multi-stage training with progressive unfreezing"""
    print("\n STARTING MULTI-STAGE TRAINING")
    print("="*50)
    
    all_histories = []
    
    # Stage 1: Initial training with frozen base
    print("\n STAGE 1: Initial Training (Frozen Base)")
    print("-" * 40)
    
    model = build_enhanced_model(num_classes=len(train_gen.classes), stage=1)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            'cancer_type': 'categorical_crossentropy', 
            'is_cancerous': 'binary_crossentropy'
        },
        metrics={
            'cancer_type': ['accuracy'],
            'is_cancerous': ['accuracy']
        },
        loss_weights={'cancer_type': 2.0, 'is_cancerous': 1.0}
    )
    
    progress_callback = DetailedProgressCallback()
    progress_callback.set_stage("STAGE 1")
    
    callbacks_stage1 = [
        progress_callback,
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        SafeModelCheckpoint("stage1_weights.h5", monitor='val_loss', save_best_only=True)
    ]
    
    try:
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=8,
            callbacks=callbacks_stage1,
            verbose=0
        )
        all_histories.append(history1)
        print(" Stage 1 completed successfully")
    except Exception as e:
        print(f" Stage 1 failed: {e}")
        return None, None
    
    # Stage 2: Progressive unfreezing
    print("\n STAGE 2: Progressive Unfreezing")
    print("-" * 40)
    
    # Rebuild model with more unfrozen layers
    model = build_enhanced_model(num_classes=len(train_gen.classes), stage=2)
    
    try:
        model.load_weights("stage1_weights.h5")
        print(" Loaded Stage 1 weights")
    except:
        print(" Could not load Stage 1 weights, continuing with random initialization")
    
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss={
            'cancer_type': 'categorical_crossentropy', 
            'is_cancerous': 'binary_crossentropy'
        },
        metrics={
            'cancer_type': ['accuracy'],
            'is_cancerous': ['accuracy']
        },
        loss_weights={'cancer_type': 2.0, 'is_cancerous': 1.0}
    )
    
    progress_callback.set_stage("STAGE 2")
    
    callbacks_stage2 = [
        progress_callback,
        ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        SafeModelCheckpoint("stage2_weights.h5", monitor='val_loss', save_best_only=True),
        LearningRateScheduler(cosine_decay_with_warmup, verbose=1)
    ]
    
    try:
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=12,
            callbacks=callbacks_stage2,
            verbose=0
        )
        all_histories.append(history2)
        print(" Stage 2 completed successfully")
    except Exception as e:
        print(f" Stage 2 failed: {e}")
        return model, all_histories
    
    # Stage 3: Fine-tuning with full model
    print("\n STAGE 3: Fine-tuning (Full Model)")
    print("-" * 40)
    
    # Rebuild with fully unfrozen model
    model = build_enhanced_model(num_classes=len(train_gen.classes), stage=3)
    
    try:
        model.load_weights("stage2_weights.h5")
        print(" Loaded Stage 2 weights")
    except:
        print(" Could not load Stage 2 weights, using current model")
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss={
            'cancer_type': 'categorical_crossentropy', 
            'is_cancerous': 'binary_crossentropy'
        },
        metrics={
            'cancer_type': ['accuracy'],
            'is_cancerous': ['accuracy']
        },
        loss_weights={'cancer_type': 2.0, 'is_cancerous': 1.0}
    )
    
    progress_callback.set_stage("STAGE 3")
    
    callbacks_stage3 = [
        progress_callback,
        ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.2, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        SafeModelCheckpoint("final_model_weights.h5", monitor='val_loss', save_best_only=True)
    ]
    
    try:
        history3 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            callbacks=callbacks_stage3,
            verbose=0
        )
        all_histories.append(history3)
        print(" Stage 3 completed successfully")
    except Exception as e:
        print(f" Stage 3 failed: {e}")
    
    return model, all_histories

#    Execute Training  
final_model, training_histories = multi_stage_training()

#    Comprehensive Evaluation  
def comprehensive_evaluation(model, val_generator):
    """Perform comprehensive model evaluation"""
    print("\n COMPREHENSIVE EVALUATION")
    print("="*50)
    
    # Load best weights
    try:
        model.load_weights("final_model_weights.h5")
        print(" Loaded best model weights for evaluation")
    except:
        print(" Using current model weights for evaluation")
    
    # Collect validation data
    val_imgs, y_true = [], {'cancer_type': [], 'is_cancerous': []}
    
    print(" Collecting validation predictions...")
    for i in range(len(val_generator)):
        X, y = val_generator[i]
        val_imgs.append(X)
        y_true['cancer_type'].extend(np.argmax(y['cancer_type'], axis=1))
        y_true['is_cancerous'].extend(y['is_cancerous'])
    
    X_all = np.concatenate(val_imgs)
    y_pred = model.predict(X_all, batch_size=32, verbose=0)
    
    y_pred_classes = {
        'cancer_type': np.argmax(y_pred[0], axis=1),
        'is_cancerous': (y_pred[1] > 0.5).astype(int).flatten()
    }
    
    # Calculate comprehensive metrics
    type_accuracy = accuracy_score(y_true['cancer_type'], y_pred_classes['cancer_type'])
    cancer_accuracy = accuracy_score(y_true['is_cancerous'], y_pred_classes['is_cancerous'])
    
    try:
        f1 = f1_score(y_true['is_cancerous'], y_pred_classes['is_cancerous'], average='weighted')
    except:
        f1 = 0.0
    
    try:
        auc = roc_auc_score(y_true['is_cancerous'], y_pred[1])
    except:
        auc = 0.5
    
    # Per-class analysis
    print("\n DETAILED CLASSIFICATION REPORT")
    print("-" * 50)
    
    # Cancer type classification report
    print("\n Cancer Type Classification:")
    try:
        type_report = classification_report(
            y_true['cancer_type'], 
            y_pred_classes['cancer_type'],
            target_names=val_generator.classes,
            output_dict=True
        )
        
        for class_name, metrics in type_report.items():
            if isinstance(metrics, dict):
                print(f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    except Exception as e:
        print(f" Could not generate type classification report: {e}")
        type_report = {}
    
    # Cancer detection report
    print("\n🎯 Cancer Detection:")
    try:
        cancer_report = classification_report(
            y_true['is_cancerous'], 
            y_pred_classes['is_cancerous'],
            target_names=['Non-Cancerous', 'Cancerous'],
            output_dict=True
        )
        
        for class_name, metrics in cancer_report.items():
            if isinstance(metrics, dict):
                print(f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    except Exception as e:
        print(f" Could not generate cancer detection report: {e}")
        cancer_report = {}
    
    return {
        'type_accuracy': type_accuracy,
        'cancer_accuracy': cancer_accuracy,
        'f1_score': f1,
        'auc_roc': auc,
        'type_report': type_report,
        'cancer_report': cancer_report
    }

#  Final Evaluation 
if final_model is not None:
    evaluation_results = comprehensive_evaluation(final_model, val_gen)
    
    #   Results Display  
    print("\n" + "="*60)
    print(" FINAL EVALUATION SUMMARY")
    print("="*60)
    print(f" Cancer Type Classification Accuracy: {evaluation_results['type_accuracy']:.4f}")
    print(f" Cancer Detection Accuracy: {evaluation_results['cancer_accuracy']:.4f}")
    print(f" Weighted F1 Score: {evaluation_results['f1_score']:.4f}")
    print(f" AUC-ROC Score: {evaluation_results['auc_roc']:.4f}")
    
    # Performance assessment
    target_achieved = evaluation_results['type_accuracy'] >= 0.85
    print(f"\n Target Accuracy (≥0.85): {' ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    if target_achieved:
        print(" Congratulations! Your model has reached the target performance!")
    else:
        improvement_needed = 0.85 - evaluation_results['type_accuracy']
        print(f" Improvement needed: {improvement_needed:.1%}")
        # print("\n Suggestions for further improvement:")
        # print("   • Increase training data")
        # print("   • Try ensemble methods")
        # print("   • Experiment with other architectures (ConvNeXT, Vision Transformer)")
        # print("   • Implement test-time augmentation")
        # print("   • Use pseudo-labeling with unlabeled data")
    
    print("\n" + "="*60)
    
    # Save final results
    try:
        final_model.save_weights("final_trained_model.h5")
        print(" Final model weights saved as 'final_trained_model.h5'")
    except Exception as e:
        print(f" Could not save final model: {e}")

else:
    print(" Training failed completely. Please check your data and configuration.")

print("\n Training and evaluation completed!")
