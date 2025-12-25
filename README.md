# Qwen 1.7B LoRA Fine-tuning - Hyperparameter Guide

## Overview
This guide explains all hyperparameters in the training script and how adjusting them affects model training, memory usage, and final performance.

---

## Table of Contents
- [LoRA Hyperparameters](#lora-hyperparameters)
- [Training Hyperparameters](#training-hyperparameters)
- [Model & Quantization](#model--quantization)
- [Optimization Tips](#optimization-tips)
- [Common Scenarios](#common-scenarios)

---

## LoRA Hyperparameters

### `LORA_R` (Rank) - Default: 16
**What it is:** The dimensionality of the low-rank matrices used in LoRA adaptation.

**Effect on Training:**
- **Lower values (4-8):**
  - ✅ Faster training
  - ✅ Less memory usage
  - ✅ Fewer trainable parameters
  - ❌ Lower model capacity
  - ❌ May underfit complex tasks
  
- **Higher values (32-64):**
  - ✅ Better performance on complex tasks
  - ✅ More expressive model
  - ❌ Slower training
  - ❌ More memory required
  - ❌ Risk of overfitting on small datasets

**Recommendation:**
- Simple tasks: `r=8`
- Medium complexity: `r=16` (default)
- Complex tasks: `r=32`
- Very complex/large datasets: `r=64`

---

### `LORA_ALPHA` (Scaling Factor) - Default: 32
**What it is:** Scaling factor for LoRA updates, typically set to 2× the rank.

**Effect on Training:**
- **Formula:** `effective_learning_rate = LORA_ALPHA / LORA_R × base_learning_rate`
- **Higher alpha:** Stronger LoRA influence, faster adaptation
- **Lower alpha:** More conservative updates, more stable training

**Relationship with LORA_R:**
```
Common ratios:
- LORA_ALPHA = LORA_R       → Conservative (1:1)
- LORA_ALPHA = 2 × LORA_R   → Balanced (2:1) [RECOMMENDED]
- LORA_ALPHA = 4 × LORA_R   → Aggressive (4:1)
```

**Recommendation:**
- Keep `LORA_ALPHA = 2 × LORA_R` for most cases
- Adjust only if experiencing instability or slow convergence

---

### `LORA_DROPOUT` - Default: 0.05
**What it is:** Dropout probability applied to LoRA layers during training.

**Effect on Training:**
- **Lower values (0.0-0.05):**
  - Better for small datasets
  - Faster convergence
  - Risk of overfitting
  
- **Higher values (0.1-0.2):**
  - Better regularization
  - Prevents overfitting
  - Slightly slower convergence

**Recommendation:**
- Small datasets (<100 examples): `0.0-0.05`
- Medium datasets (100-1000): `0.05-0.1`
- Large datasets (>1000): `0.1-0.2`

---

### `LORA_TARGET_MODULES` - Default: `["q_proj", "k_proj", "v_proj", "o_proj"]`
**What it is:** Which model layers to apply LoRA adapters to.

**Options:**
```python
# Attention only (least parameters, fastest)
["q_proj", "k_proj", "v_proj", "o_proj"]

# Attention + MLP projection (balanced)
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# All linear layers (most parameters, best performance)
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
```

**Trade-offs:**
| Configuration | Trainable Params | Speed | Performance |
|--------------|------------------|-------|-------------|
| Attention only | ~8-16M | Fastest | Good |
| Attention + MLP | ~24-48M | Medium | Better |
| All layers | ~48-96M | Slowest | Best |

---

## Training Hyperparameters

### `NUM_EPOCHS` - Default: 3
**What it is:** Number of complete passes through the training dataset.

**Effect on Training:**
- **1-2 epochs:**
  - Quick training
  - May underfit
  - Good for large datasets
  
- **3-5 epochs:**
  - Balanced training
  - Good for most use cases
  
- **5+ epochs:**
  - Risk of overfitting
  - Only for large, diverse datasets
  - Longer training time

**Signs you need adjustment:**
- **Underfitting:** Loss still decreasing at end → Increase epochs
- **Overfitting:** Loss starts increasing after epoch X → Reduce epochs to X

---

### `BATCH_SIZE` - Default: 4
**What it is:** Number of examples processed simultaneously per GPU.

**Effect on Training:**
- **Smaller (1-2):**
  - ✅ Fits in limited VRAM
  - ✅ More gradient updates (noisier but can help)
  - ❌ Slower training
  - ❌ Less stable gradients
  
- **Larger (8-16):**
  - ✅ Faster training
  - ✅ More stable gradients
  - ❌ Requires more VRAM
  - ❌ May need lower learning rate

**Memory Impact:**
```
Colab T4 (15GB):
- batch_size=1: ~6GB usage
- batch_size=2: ~8GB usage
- batch_size=4: ~12GB usage
- batch_size=8: OOM (Out of Memory)
```

**Recommendation:**
- T4 GPU (Free Colab): `batch_size=2-4`
- V100/A100: `batch_size=8-16`
- Limited VRAM: `batch_size=1` + increase `GRADIENT_ACCUMULATION_STEPS`

---

### `GRADIENT_ACCUMULATION_STEPS` - Default: 4
**What it is:** Number of forward/backward passes before updating weights.

**Effective batch size = `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS`**

**Effect on Training:**
- Allows simulating large batch sizes without VRAM increase
- Higher values: More stable gradients but slower iteration
- Lower values: Faster but noisier training

**Example configurations:**
```python
# Config A: Small actual, large effective
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
# Effective batch = 16, VRAM usage = small

# Config B: Large actual, small effective
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
# Effective batch = 16, VRAM usage = large
```

**Recommendation:**
- Keep effective batch size between 16-32 for stable training
- Adjust based on VRAM availability

---

### `LEARNING_RATE` - Default: 2e-4
**What it is:** Step size for weight updates during optimization.

**Effect on Training:**
- **Too low (1e-5):**
  - Very slow convergence
  - May need many more epochs
  - Safe but inefficient
  
- **Optimal (1e-4 to 5e-4):**
  - Good convergence speed
  - Stable training
  
- **Too high (1e-3+):**
  - Training instability
  - Loss may diverge
  - Poor final performance

**Relationship with batch size:**
```python
# Linear scaling rule
if batch_size doubles → consider increasing LR by √2

Examples:
batch_size=4:  LR=2e-4
batch_size=8:  LR=3e-4
batch_size=16: LR=4e-4
```

**Recommendation:**
- Start with `2e-4`
- If loss plateaus early: Increase to `3e-4` or `5e-4`
- If training unstable: Decrease to `1e-4`

---

### `MAX_LENGTH` - Default: 2048
**What it is:** Maximum sequence length (tokens) for training examples.

**Effect on Training:**
- **VRAM Usage:** Memory scales quadratically with sequence length
- **Shorter (512-1024):**
  - ✅ Lower memory usage
  - ✅ Faster training
  - ❌ May truncate long documents
  
- **Longer (2048-4096):**
  - ✅ Handles longer contexts
  - ❌ Much higher VRAM usage
  - ❌ Slower training

**Memory impact per token:**
```
512 tokens:  ~4GB VRAM
1024 tokens: ~8GB VRAM
2048 tokens: ~12GB VRAM
4096 tokens: ~20GB+ VRAM (requires A100)
```

**Recommendation:**
- Analyze your data first: Check typical document lengths
- Set `MAX_LENGTH` to cover 90-95% of your data
- For Typst code generation: `2048` is usually sufficient

---

### `WARMUP_STEPS` - Default: 100
**What it is:** Number of steps to gradually increase learning rate from 0 to target.

**Effect on Training:**
- Prevents early training instability
- Helps model adapt gradually
- **Too few:** May have unstable start
- **Too many:** Wastes training time

**Recommendation:**
```python
# Rule of thumb: 5-10% of total training steps
total_steps = (dataset_size / effective_batch_size) * NUM_EPOCHS
warmup_steps = total_steps * 0.05  # 5% of total
```

---

### `WEIGHT_DECAY` - Default: 0.01
**What it is:** L2 regularization strength to prevent overfitting.

**Effect on Training:**
- **0.0:** No regularization (may overfit)
- **0.01:** Light regularization (good default)
- **0.1:** Strong regularization (may underfit)

**Recommendation:**
- Small datasets: `0.01-0.05`
- Large datasets: `0.001-0.01`
- If overfitting: Increase to `0.05` or `0.1`

---

## Model & Quantization

### `USE_4BIT` - Default: True
**What it is:** Whether to use 4-bit quantization (QLoRA).

**4-bit Quantization (True):**
- ✅ ~75% memory reduction
- ✅ Fits on free Colab T4 GPU
- ✅ Minimal performance loss (<2%)
- ❌ Slightly slower training

**Full precision (False):**
- ✅ Slightly better final performance
- ✅ Faster training
- ❌ Requires ~4x more VRAM
- ❌ Needs A100 or multiple GPUs

**Recommendation:**
- Colab Free (T4): `USE_4BIT = True` (required)
- Colab Pro (V100): `USE_4BIT = True` (recommended)
- A100 GPU: `USE_4BIT = False` (for best performance)

---

## Optimization Tips

### Memory Optimization (OOM Errors)
If you encounter Out-of-Memory errors:
```python
# Priority 1: Reduce batch size
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16  # Maintain effective batch

# Priority 2: Reduce sequence length
MAX_LENGTH = 1024  # or 512

# Priority 3: Reduce LoRA rank
LORA_R = 8
LORA_ALPHA = 16

# Priority 4: Target fewer modules
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Minimum
```

### Speed Optimization
To train faster:
```python
# Increase batch size (if VRAM allows)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

# Reduce sequence length
MAX_LENGTH = 1024

# Fewer epochs
NUM_EPOCHS = 2

# Use gradient checkpointing
# Add to TrainingArguments:
gradient_checkpointing=True
```

### Quality Optimization
For best model performance:
```python
# Higher LoRA rank
LORA_R = 32
LORA_ALPHA = 64

# More modules
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# More epochs (but watch for overfitting)
NUM_EPOCHS = 5

# Larger effective batch
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4  # Effective: 32
```

---

## Common Scenarios

### Scenario 1: Limited VRAM (Colab Free T4)
```python
MODEL_NAME = "Qwen/Qwen-1_8B"
LORA_R = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
MAX_LENGTH = 1024
USE_4BIT = True
```

### Scenario 2: Balanced (V100 or Colab Pro)
```python
MODEL_NAME = "Qwen/Qwen-1_8B"
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 2048
USE_4BIT = True
```

### Scenario 3: Maximum Quality (A100)
```python
MODEL_NAME = "Qwen/Qwen-1_8B"
LORA_R = 64
LORA_ALPHA = 128
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 4096
USE_4BIT = False
```

### Scenario 4: Fast Iteration (Testing)
```python
MODEL_NAME = "Qwen/Qwen-1_8B"
LORA_R = 8
LORA_ALPHA = 16
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_LENGTH = 512
NUM_EPOCHS = 1
USE_4BIT = True
```

---

## Monitoring Training

### Key Metrics to Watch

**Training Loss:**
- Should consistently decrease
- Typical range: 2.0 → 0.5 over training
- If stuck: Increase learning rate
- If erratic: Decrease learning rate or increase batch size

**Signs of Overfitting:**
- Loss stops decreasing
- Loss starts increasing
- Model memorizes training data
- **Solution:** Reduce epochs, increase dropout, add weight decay

**Signs of Underfitting:**
- Loss remains high
- Loss still decreasing at end
- Poor generation quality
- **Solution:** Increase epochs, increase model capacity (LORA_R), lower regularization

---

## Quick Reference Table

| Hyperparameter | Impact | Adjust Up If... | Adjust Down If... |
|----------------|--------|-----------------|-------------------|
| LORA_R | Model capacity | Underfitting | Out of memory |
| NUM_EPOCHS | Training time | Loss decreasing | Overfitting |
| BATCH_SIZE | Speed/VRAM | Have memory | OOM errors |
| LEARNING_RATE | Convergence speed | Slow progress | Training unstable |
| MAX_LENGTH | Context size | Truncating data | OOM errors |
| LORA_DROPOUT | Regularization | Overfitting | Underfitting |

---

## Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen Documentation](https://github.com/QwenLM/Qwen)
- [PEFT Documentation](https://huggingface.co/docs/peft)

---

## Troubleshooting

### OOM (Out of Memory)
1. Set `BATCH_SIZE = 1`
2. Set `MAX_LENGTH = 512`
3. Set `LORA_R = 8`
4. Enable `USE_4BIT = True`

### Training Too Slow
1. Increase `BATCH_SIZE` (if VRAM allows)
2. Reduce `MAX_LENGTH`
3. Reduce `NUM_EPOCHS`
4. Use gradient checkpointing

### Poor Results
1. Increase `LORA_R` to 32
2. Increase `NUM_EPOCHS` to 5
3. Check data quality and format
4. Add more diverse training examples

### Training Unstable (Loss Spikes)
1. Reduce `LEARNING_RATE` to 1e-4
2. Increase `WARMUP_STEPS` to 200
3. Increase effective batch size
4. Check for corrupted data

---

## Contact & Support
For issues specific to this training script, check the code comments or modify parameters based on this guide.

---

## Inference - Using the Trained Model

After training your model, use the `inference_qwen_lora.py` script to generate Typst code from new inputs.

### Quick Start

**Command Line Usage:**
```bash
# Basic inference - output to console
python inference_qwen_lora.py --input your_input.json

# Save output to file
python inference_qwen_lora.py --input your_input.json --output generated_output.json

# Custom parameters
python inference_qwen_lora.py --input input.json --temperature 0.5 --max-tokens 3000
```

### Input JSON Format

Your input JSON should follow the same structure as training data:

**Single Input:**
```json
{
  "input": {
    "starting_page": 1,
    "academic_year": "2025-26",
    "department": "B.E/Dept of CSE/BNMIT",
    "project_title": "Your Project Title",
    "contents_txt": "Your chapter content goes here..."
  }
}
```

**Multiple Inputs (Batch Processing):**
```json
[
  {
    "input": {
      "starting_page": 1,
      "academic_year": "2025-26",
      "department": "B.E/Dept of CSE/BNMIT",
      "project_title": "Project 1",
      "contents_txt": "Content 1..."
    }
  },
  {
    "input": {
      "starting_page": 10,
      "academic_year": "2025-26",
      "department": "B.E/Dept of CSE/BNMIT",
      "project_title": "Project 2",
      "contents_txt": "Content 2..."
    }
  }
]
```

### Python Script Usage

For programmatic access or integration into your workflow:

```python
from inference_qwen_lora import load_model_and_tokenizer, inference_from_dict

# Load model once (reuse for multiple generations)
model, tokenizer = load_model_and_tokenizer(
    base_model_name="Qwen/Qwen-1_8B",
    adapter_path="./qwen-lora-finetuned",
    use_4bit=True
)

# Single inference
input_data = {
    "starting_page": 1,
    "academic_year": "2025-26",
    "department": "B.E/Dept of CSE/BNMIT",
    "project_title": "Machine Learning Project",
    "contents_txt": "Chapter content goes here..."
}

output = inference_from_dict(model, tokenizer, input_data)
print(output)

# Save to Typst file
with open("output.typ", "w", encoding="utf-8") as f:
    f.write(output)
```

### Generation Parameters

Control the output quality and style using these parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--temperature` | 0.7 | 0.1-2.0 | Controls randomness. Lower=deterministic, Higher=creative |
| `--max-tokens` | 2048 | 512-4096 | Maximum length of generated output |
| `--top-p` | 0.95 | 0.1-1.0 | Nucleus sampling. Lower=focused, Higher=diverse |
| `--top-k` | 50 | 1-100 | Top-k sampling for token selection |
| `--repetition-penalty` | 1.1 | 1.0-2.0 | Penalize repeated phrases |

**Temperature Guidelines:**
- **0.1-0.3**: Very deterministic, consistent formatting (recommended for production)
- **0.5-0.7**: Balanced output with slight variation
- **0.8-1.2**: More creative, varied output (may deviate from training style)
- **1.5-2.0**: Highly creative (not recommended for structured output)

### Command Line Options

```bash
python inference_qwen_lora.py \
  --input input.json \              # Required: Input JSON file
  --output output.json \            # Optional: Output file (default: console)
  --model ./qwen-lora-finetuned \   # Optional: LoRA adapter path
  --base-model Qwen/Qwen-1_8B \     # Optional: Base model name
  --max-tokens 2048 \               # Optional: Max generation length
  --temperature 0.7 \               # Optional: Sampling temperature
  --no-4bit                         # Optional: Disable quantization (needs more VRAM)
```

### Batch Processing Example

Process multiple documents efficiently:

```python
from inference_qwen_lora import load_model_and_tokenizer, inference_from_json_file

# Load model once
model, tokenizer = load_model_and_tokenizer(
    "Qwen/Qwen-1_8B",
    "./qwen-lora-finetuned"
)

# Process multiple files
input_files = ["chapter1.json", "chapter2.json", "chapter3.json"]

for input_file in input_files:
    output_file = input_file.replace(".json", "_output.json")
    inference_from_json_file(model, tokenizer, input_file, output_file)
    print(f"✓ Processed: {input_file} → {output_file}")
```

### Google Colab Inference

**Step 1: Upload files to Colab**
```python
from google.colab import files
uploaded = files.upload()  # Upload your input.json
```

**Step 2: Run inference**
```bash
!python inference_qwen_lora.py --input input.json --output result.json
```

**Step 3: Download results**
```python
from google.colab import files
files.download('result.json')
```

### Output Format

The script outputs JSON with both input and generated code:

```json
[
  {
    "input": {
      "starting_page": 1,
      "academic_year": "2025-26",
      "department": "B.E/Dept of CSE/BNMIT",
      "project_title": "Project Title",
      "contents_txt": "Content..."
    },
    "generated_output": "#page(\n  paper: \"a4\",\n  margin: (top: 1in, bottom: 1in, left: 1.25in, right: 1in),\n  ..."
  }
]
```

To extract just the Typst code:
```python
import json

with open("result.json", "r") as f:
    data = json.load(f)

# Save first result as .typ file
with open("output.typ", "w", encoding="utf-8") as f:
    f.write(data[0]["generated_output"])
```

### Troubleshooting Inference

**Problem: Out of Memory**
```bash
# Solution: Enable 4-bit quantization (default) or reduce max tokens
python inference_qwen_lora.py --input input.json --max-tokens 1024
```

**Problem: Poor Quality Output**
```bash
# Solution 1: Lower temperature for more consistency
python inference_qwen_lora.py --input input.json --temperature 0.3

# Solution 2: Increase max tokens if output is truncated
python inference_qwen_lora.py --input input.json --max-tokens 3000
```

**Problem: Output too repetitive**
```bash
# Solution: Increase repetition penalty
python inference_qwen_lora.py --input input.json --repetition-penalty 1.5
```

**Problem: Model not found**
```bash
# Solution: Specify correct paths
python inference_qwen_lora.py \
  --input input.json \
  --model ./qwen-lora-finetuned \
  --base-model Qwen/Qwen-1_8B
```

### Performance Tips

1. **Load Model Once**: In Python scripts, load the model once and reuse it for multiple inferences
2. **Batch Processing**: Process multiple inputs in one session to amortize model loading time
3. **GPU Usage**: Always use GPU for faster inference (especially on Colab)
4. **4-bit Quantization**: Keep enabled unless you need absolute best quality and have enough VRAM
5. **Token Limits**: Set `max-tokens` based on your typical output length to avoid unnecessary computation

### Integration Examples

**Flask API Server:**
```python
from flask import Flask, request, jsonify
from inference_qwen_lora import load_model_and_tokenizer, inference_from_dict

app = Flask(__name__)
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen-1_8B", "./qwen-lora-finetuned")

@app.route('/generate', methods=['POST'])
def generate():
    input_data = request.json
    output = inference_from_dict(model, tokenizer, input_data)
    return jsonify({"generated_output": output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Streamlit Web App:**
```python
import streamlit as st
from inference_qwen_lora import load_model_and_tokenizer, inference_from_dict

@st.cache_resource
def load_model():
    return load_model_and_tokenizer("Qwen/Qwen-1_8B", "./qwen-lora-finetuned")

model, tokenizer = load_model()

st.title("Typst Code Generator")
project_title = st.text_input("Project Title")
contents = st.text_area("Chapter Contents")

if st.button("Generate"):
    input_data = {
        "starting_page": 1,
        "academic_year": "2025-26",
        "department": "B.E/Dept of CSE/BNMIT",
        "project_title": project_title,
        "contents_txt": contents
    }
    output = inference_from_dict(model, tokenizer, input_data)
    st.code(output, language="typst")
```

---

## Complete Workflow Summary

### 1. Training Phase
```bash
# Prepare your training data (105 samples in JSON format)
# Configure hyperparameters in train_qwen_lora.py
# Run training
python train_qwen_lora.py
```

### 2. Inference Phase
```bash
# Prepare input JSON
# Run inference
python inference_qwen_lora.py --input new_input.json --output result.json

# Extract Typst code
# Use generated code in your Typst documents
```

### 3. Continued Training (Optional)
```bash
# Add more samples to your JSON file
# Adjust learning rate to 1e-4 (lower)
# Retrain from checkpoint
python train_qwen_lora.py  # Will load existing adapter if found
```
