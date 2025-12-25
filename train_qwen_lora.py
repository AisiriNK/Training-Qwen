"""
Qwen 1.7B Fine-tuning Script with LoRA
Designed for Google Colab
Training data format: JSON with input/output pairs
"""

# ============================================================================
# SECTION 1: Install Dependencies (Run in Colab cell)
# ============================================================================
"""
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Authenticate with Hugging Face (REQUIRED for Qwen models)
from huggingface_hub import login
login()  # Enter your HF token when prompted

# Install compatible versions
!pip install -q transformers==4.46.0
!pip install -q peft==0.13.0
!pip install -q datasets==3.0.0
!pip install -q accelerate==1.0.0
!pip install -q bitsandbytes==0.44.0
!pip install -q trl==0.11.0
!pip install -q rouge-score==0.1.2
!pip install -q sacrebleu==2.4.0
!pip install -q sentencepiece==0.2.0
!pip install -q fsspec==2025.3.0
"""

# ============================================================================
# SECTION 2: Import Libraries
# ============================================================================
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
import os
from typing import Dict, List, Tuple
import math

# ============================================================================
# SECTION 3: Configuration
# ============================================================================
class Config:
    # Model configuration
    MODEL_NAME = "Qwen/Qwen-1_8B"  # Qwen 1.7B model
    USE_AUTH_TOKEN = True  # Enable HuggingFace authentication
    
    # LoRA configuration (Optimized for 105 samples)
    LORA_R = 16  # Rank of LoRA matrices - balanced for small dataset
    LORA_ALPHA = 32  # Scaling factor (2x rank)
    LORA_DROPOUT = 0.1  # Higher dropout to prevent overfitting on small dataset
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Attention layers
    
    # Training configuration (Optimized for 105 samples)
    NUM_EPOCHS = 5  # More epochs for small dataset
    BATCH_SIZE = 2  # Smaller batch for better gradient updates
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 16
    LEARNING_RATE = 2e-4  # Standard learning rate
    MAX_LENGTH = 2048  # Sufficient for Typst code generation
    
    # Regularization (Important for small datasets)
    WEIGHT_DECAY = 0.01  # L2 regularization
    WARMUP_STEPS = 50  # ~10% of total steps
    
    # Quantization (for memory efficiency)
    USE_4BIT = True  # Set to False if you have enough VRAM
    
    # Data paths - SET THESE BEFORE RUNNING
    # After uploading your JSON file in Colab, update the path below
    TRAIN_DATA_PATH = "/content/training_pair2.3.5.json"  # Change this to your uploaded file path
    OUTPUT_DIR = "/content/qwen-lora-adapter"  # Adapter will be saved here (same location as data)

config = Config()

# ============================================================================
# SECTION 4: Data Loading and Preprocessing
# ============================================================================
def load_training_data(file_path: str) -> List[Dict]:
    """Load training data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def format_training_example(example: Dict) -> str:
    """
    Format the training example into a prompt-completion format
    Converts the input fields into a structured prompt
    """
    input_data = example['input']
    output_data = example['output']
    
    # Create a structured prompt
    prompt = f"""### Task: Generate Typst document formatting code

### Input Specifications:
- Starting Page: {input_data['starting_page']}
- Academic Year: {input_data['academic_year']}
- Department: {input_data['department']}
- Project Title: {input_data['project_title']}

### Content:
{input_data['contents_txt']}

### Output (Typst Code):
{output_data}"""
    
    return prompt

def prepare_dataset(data: List[Dict], tokenizer, validation_split: float = 0.1) -> Tuple[Dataset, Dataset, List[Dict]]:
    """
    Prepare and tokenize the dataset with train/validation split
    
    Returns:
        train_dataset, val_dataset, val_raw_data
    """
    # Split data into train and validation
    split_idx = int(len(data) * (1 - validation_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Dataset split: {len(train_data)} training, {len(val_data)} validation")
    
    # Format all examples
    train_texts = [format_training_example(example) for example in train_data]
    val_texts = [format_training_example(example) for example in val_data]
    
    # Tokenize function
    def tokenize_function(examples):
        # Tokenize the texts
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Create and tokenize train dataset
    train_dataset_dict = {"text": train_texts}
    train_dataset = Dataset.from_dict(train_dataset_dict)
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )
    
    # Create and tokenize validation dataset
    val_dataset_dict = {"text": val_texts}
    val_dataset = Dataset.from_dict(val_dataset_dict)
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset"
    )
    
    return train_dataset, val_dataset, val_data

# ============================================================================
# SECTION 5: Model Loading and LoRA Setup
# ============================================================================
def setup_model_and_tokenizer():
    """Load model and tokenizer with quantization and LoRA"""
    
    # Quantization config for 4-bit training (memory efficient)
    if config.USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,  # Important for causal LM
        token=config.USE_AUTH_TOKEN  # Add authentication
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config if config.USE_4BIT else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        token=config.USE_AUTH_TOKEN  # Add authentication
    )
    
    # Prepare model for k-bit training
    if config.USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Add LoRA adapters to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

# ============================================================================
# SECTION 6: Evaluation Functions
# ============================================================================
def calculate_perplexity(model, dataset, tokenizer) -> float:
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    print("Calculating perplexity...")
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs = {k: torch.tensor(v).unsqueeze(0).to(model.device) 
                     for k, v in dataset[i].items() if k in ['input_ids', 'attention_mask', 'labels']}
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Count non-padding tokens
            mask = inputs['labels'] != tokenizer.pad_token_id
            num_tokens = mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def generate_sample_outputs(model, tokenizer, val_data: List[Dict], num_samples: int = 3) -> List[Dict]:
    """Generate outputs for sample validation data"""
    model.eval()
    results = []
    
    print(f"\nGenerating {num_samples} sample outputs for evaluation...")
    
    for i, example in enumerate(val_data[:num_samples]):
        input_data = example['input']
        expected_output = example['output']
        
        # Format prompt (same as training)
        prompt = f"""### Task: Generate Typst document formatting code

### Input Specifications:
- Starting Page: {input_data['starting_page']}
- Academic Year: {input_data['academic_year']}
- Department: {input_data['department']}
- Project Title: {input_data['project_title']}

### Content:
{input_data['contents_txt']}

### Output (Typst Code):
"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_output = generated_text[len(prompt):].strip()
        
        results.append({
            'sample_id': i + 1,
            'expected': expected_output,
            'generated': generated_output
        })
        
        print(f"  Sample {i+1}/{num_samples} generated")
    
    return results

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate BLEU and ROUGE scores"""
    
    # Initialize scorers
    bleu = BLEU()
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    print("\nCalculating BLEU and ROUGE scores...")
    
    for result in results:
        expected = result['expected']
        generated = result['generated']
        
        # BLEU score
        bleu_score = bleu.sentence_score(generated, [expected]).score
        bleu_scores.append(bleu_score)
        
        # ROUGE scores
        rouge_scores = rouge.score(expected, generated)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
    
    return {
        'bleu': np.mean(bleu_scores),
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
        'bleu_scores': bleu_scores,
        'rouge1_scores': rouge1_scores,
        'rouge2_scores': rouge2_scores,
        'rougeL_scores': rougeL_scores
    }

def evaluate_model(model, tokenizer, val_dataset, val_data: List[Dict]) -> Dict:
    """
    Comprehensive model evaluation
    
    Returns dictionary with all evaluation metrics
    """
    print("\n" + "="*80)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*80)
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, val_dataset, tokenizer)
    
    # Generate sample outputs
    num_samples = min(3, len(val_data))
    sample_results = generate_sample_outputs(model, tokenizer, val_data, num_samples)
    
    # Calculate BLEU and ROUGE scores
    metrics = calculate_metrics(sample_results)
    
    # Compile all results
    evaluation_results = {
        'perplexity': perplexity,
        'avg_bleu': metrics['bleu'],
        'avg_rouge1': metrics['rouge1'],
        'avg_rouge2': metrics['rouge2'],
        'avg_rougeL': metrics['rougeL'],
        'sample_outputs': sample_results
    }
    
    return evaluation_results

def print_evaluation_results(results: Dict):
    """Print formatted evaluation results"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  • Perplexity:          {results['perplexity']:.4f}")
    print(f"  • BLEU Score:          {results['avg_bleu']:.4f} (0-100 scale)")
    print(f"  • ROUGE-1 F1:          {results['avg_rouge1']:.4f}")
    print(f"  • ROUGE-2 F1:          {results['avg_rouge2']:.4f}")
    print(f"  • ROUGE-L F1:          {results['avg_rougeL']:.4f}")
    
    print(f"\n📝 Sample Outputs:")
    for sample in results['sample_outputs']:
        print(f"\n  --- Sample {sample['sample_id']} ---")
        print(f"  Expected (first 200 chars):")
        print(f"  {sample['expected'][:200]}...")
        print(f"\n  Generated (first 200 chars):")
        print(f"  {sample['generated'][:200]}...")
        print(f"  {'-'*76}")
    
    print(f"\n💡 Interpretation Guide:")
    print(f"  • Perplexity: Lower is better (typically 5-50 for fine-tuned models)")
    print(f"  • BLEU: 0-100 scale, >30 is good, >50 is excellent for code generation")
    print(f"  • ROUGE: 0-1 scale, >0.3 is decent, >0.5 is good for text similarity")
    
    print("\n" + "="*80)

# ============================================================================
# SECTION 7: Training Setup
# ============================================================================
def setup_training():
    """Setup and execute training"""
    
    print("="*80)
    print("QWEN 1.7B LORA FINE-TUNING")
    print("="*80)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare data
    print("\nLoading training data...")
    raw_data = load_training_data(config.TRAIN_DATA_PATH)
    print(f"Loaded {len(raw_data)} training examples")
    
    # Prepare dataset with train/val split
    print("Preparing dataset...")
    train_dataset, val_dataset, val_data = prepare_dataset(raw_data, tokenizer, validation_split=0.1)
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        fp16=True,  # Mixed precision training
        save_strategy="epoch",
        logging_steps=5,  # Log more frequently for small dataset
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        save_total_limit=3,  # Keep more checkpoints to compare
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        report_to="none",  # Disable wandb, tensorboard, etc.
        remove_unused_columns=False,
        # Additional settings for small dataset
        load_best_model_at_end=False,
        save_safetensors=True,
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\nStarting training...")
    print(f"Total training steps: {len(train_dataset) * config.NUM_EPOCHS // (config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS)}")
    trainer.train()
    
    # Save the final model
    print("\nSaving model...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print(f"Model saved to: {config.OUTPUT_DIR}")
    print("="*80)
    
    # Evaluate the model
    evaluation_results = evaluate_model(model, tokenizer, val_dataset, val_data)
    print_evaluation_results(evaluation_results)
    
    # Save evaluation results
    eval_output_path = os.path.join(config.OUTPUT_DIR, "evaluation_results.json")
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        # Convert sample outputs to serializable format
        serializable_results = {
            'perplexity': evaluation_results['perplexity'],
            'avg_bleu': evaluation_results['avg_bleu'],
            'avg_rouge1': evaluation_results['avg_rouge1'],
            'avg_rouge2': evaluation_results['avg_rouge2'],
            'avg_rougeL': evaluation_results['avg_rougeL'],
            'sample_outputs': evaluation_results['sample_outputs']
        }
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Evaluation results saved to: {eval_output_path}")
    
    return model, tokenizer

# ============================================================================
# SECTION 8: Inference Function (For Testing)
# ============================================================================
def test_model(model, tokenizer, test_input: Dict):
    """Test the fine-tuned model with a sample input"""
    
    # Format the input
    prompt = f"""### Task: Generate Typst document formatting code

### Input Specifications:
- Starting Page: {test_input['starting_page']}
- Academic Year: {test_input['academic_year']}
- Department: {test_input['department']}
- Project Title: {test_input['project_title']}

### Content:
{test_input['contents_txt']}

### Output (Typst Code):
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    print("Generating output...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    generated_output = generated_text[len(prompt):]
    
    return generated_output

# ============================================================================
# SECTION 9: Main Execution
# ============================================================================
if __name__ == "__main__":
    # Check if running on GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run training
    model, tokenizer = setup_training()
    
    # Test with a sample (optional)
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    
    test_input = {
        "starting_page": 1,
        "academic_year": "2025-26",
        "department": "B.E/Dept of CSE/BNMIT",
        "project_title": "Test Project",
        "contents_txt": "This is a test content for the model."
    }
    
    output = test_model(model, tokenizer, test_input)
    print("\nGenerated Output:")
    print(output)

# ============================================================================
# SECTION 10: Usage Instructions for Google Colab
# ============================================================================
"""
GOOGLE COLAB SETUP INSTRUCTIONS:
================================

STEP 1: Setup Colab Environment
--------------------------------
1. Open Google Colab: https://colab.research.google.com/
2. Enable GPU: Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4)
3. Create a new notebook

STEP 2: Get Hugging Face Token
-------------------------------
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (Read access)
3. Copy the token (starts with "hf_...")

STEP 3: Upload Training Data Manually
--------------------------------------
1. In Colab, click the folder icon 📁 on the left sidebar
2. Click upload button and select training_pair2.3.5.json
3. File will be uploaded to /content/
4. Note the exact path (e.g., /content/training_pair2.3.5.json)

STEP 4: Update Script Paths
----------------------------
Before running, update these lines in the Config class:

config.TRAIN_DATA_PATH = "/content/training_pair2.3.5.json"  # Your uploaded file path
config.OUTPUT_DIR = "/content/qwen-lora-adapter"  # Adapter saves here

STEP 5: Run Training in Colab
------------------------------
# Cell 1: Setup
from huggingface_hub import login

# Authenticate with Hugging Face
print("🔑 Enter your Hugging Face token:")
login()

# Install packages
print("\n📦 Installing packages...")
!pip install -q transformers==4.46.0 peft==0.13.0 datasets==3.0.0 accelerate==1.0.0 bitsandbytes==0.44.0 trl==0.11.0 rouge-score==0.1.2 sacrebleu==2.4.0 sentencepiece==0.2.0 fsspec==2025.3.0
print("✓ Setup complete!")

# Cell 2: Run Training Script
# Copy entire train_qwen_lora.py content here (SECTIONS 2-9)
# OR upload this file and run: !python train_qwen_lora.py

TRAINING NOTES:
===============
- Training data: Manually upload JSON to /content/ via Colab file browser
- Trained adapter: Saves to /content/qwen-lora-adapter/ (same location)
- Training time: ~30-60 minutes on T4 GPU
- MacBook can sleep/shutdown - training runs on Google servers
- ⚠️ Files in /content/ are temporary - download adapter after training completes!

DOWNLOAD TRAINED ADAPTER:
=========================
After training completes, download the adapter to your computer:

# In Colab cell:
from google.colab import files
import shutil

# Zip the adapter folder
shutil.make_archive('/content/qwen-lora-adapter', 'zip', '/content/qwen-lora-adapter')

# Download the zip file
files.download('/content/qwen-lora-adapter.zip')

LOADING THE FINE-TUNED MODEL LATER:
====================================
# Upload adapter back to Colab when you need to use it

# Cell 1: Setup
from huggingface_hub import login
login()  # Enter your HF token

# Install packages
!pip install -q transformers==4.46.0 peft==0.13.0 accelerate==1.0.0 bitsandbytes==0.44.0 sentencepiece==0.2.0

# Manually upload your downloaded adapter zip to /content/ via file browser
# Then unzip it:
!unzip -q /content/qwen-lora-adapter.zip -d /content/

# Cell 2: Load Model
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    token=True
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    "/content/qwen-lora-adapter"  # Your uploaded adapter path
)

tokenizer = AutoTokenizer.from_pretrained(
    "/content/qwen-lora-adapter"
)

model.eval()
print("✓ Model loaded successfully!")

# Cell 3: Run Inference
input_data = {
    "starting_page": 1,
    "academic_year": "2025-26",
    "department": "CSE",
    "project_title": "Your Project",
    "contents_txt": "Your content here"
}

prompt = f"""### Task: Generate Typst document formatting code

### Input Specifications:
- Starting Page: {input_data['starting_page']}
- Academic Year: {input_data['academic_year']}
- Department: {input_data['department']}
- Project Title: {input_data['project_title']}

### Content:
{input_data['contents_txt']}

### Output (Typst Code):
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
result = generated[len(prompt):].strip()
print("\n=== Generated Typst Code ===")
print(result)

TROUBLESHOOTING:
================
Error: "credential propagation was unsuccessful"
Solution: Run login() and enter your HF token

Error: "Out of memory"
Solution: Runtime -> Restart runtime, ensure GPU is enabled

Error: "Cannot find file: /content/training_pair2.3.5.json"
Solution: Check file was uploaded correctly via Colab file browser (folder icon on left)

Error: "Cannot find adapter"
Solution: Ensure OUTPUT_DIR path matches where you saved the adapter

Error: "Session timeout during training"
Solution: Keep browser tab open, or use Colab Pro for longer sessions

WORKFLOW SUMMARY:
=================
1. Upload JSON → /content/ (via Colab file browser)
2. Update paths → Set TRAIN_DATA_PATH and OUTPUT_DIR in Config
3. Train model → Adapter saves to /content/qwen-lora-adapter/
4. Download adapter → Zip and download to your computer (important!)
5. Upload adapter → Re-upload when needed for inference
6. Load & use → Load adapter from /content/ and generate outputs
"""
