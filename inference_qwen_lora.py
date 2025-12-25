"""
Qwen 1.7B LoRA Inference Script
Load fine-tuned model and generate Typst code from JSON input
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
import sys

# ============================================================================
# Configuration
# ============================================================================
class InferenceConfig:
    # Model paths
    BASE_MODEL_NAME = "Qwen/Qwen-1_8B"
    LORA_ADAPTER_PATH = "./qwen-lora-finetuned"  # Path to your trained LoRA adapter
    
    # Generation parameters
    MAX_NEW_TOKENS = 2048  # Maximum length of generated output
    TEMPERATURE = 0.7  # Higher = more creative, Lower = more deterministic
    TOP_P = 0.95  # Nucleus sampling threshold
    TOP_K = 50  # Top-k sampling
    REPETITION_PENALTY = 1.1  # Penalize repetition
    DO_SAMPLE = True  # Enable sampling for diverse outputs
    
    # Quantization
    USE_4BIT = True  # Set to False for full precision

config = InferenceConfig()

# ============================================================================
# Model Loading
# ============================================================================
def load_model_and_tokenizer(base_model_name: str, adapter_path: str, use_4bit: bool = True):
    """Load the base model and LoRA adapter"""
    
    print("="*80)
    print("LOADING FINE-TUNED QWEN MODEL")
    print("="*80)
    
    # Quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load tokenizer
    print(f"Loading tokenizer from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config if use_4bit else None,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Set to evaluation mode
    model.eval()
    
    print("Model loaded successfully!")
    print("="*80)
    print()
    
    return model, tokenizer

# ============================================================================
# Input Formatting
# ============================================================================
def format_input_prompt(input_data: dict) -> str:
    """
    Format the JSON input into the same prompt structure used during training
    """
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
    
    return prompt

# ============================================================================
# Generation Function
# ============================================================================
def generate_output(model, tokenizer, input_data: dict, generation_config: dict = None) -> str:
    """
    Generate Typst code from input data
    
    Args:
        model: The loaded LoRA model
        tokenizer: The tokenizer
        input_data: Dictionary with keys: starting_page, academic_year, department, 
                    project_title, contents_txt
        generation_config: Optional custom generation parameters
    
    Returns:
        Generated Typst code as string
    """
    
    # Format the prompt
    prompt = format_input_prompt(input_data)
    
    print("Generating output...")
    print(f"Input length: {len(prompt)} characters")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Set generation parameters
    gen_config = {
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "top_k": config.TOP_K,
        "repetition_penalty": config.REPETITION_PENALTY,
        "do_sample": config.DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Override with custom config if provided
    if generation_config:
        gen_config.update(generation_config)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    generated_output = generated_text[len(prompt):].strip()
    
    print("Generation complete!")
    print(f"Output length: {len(generated_output)} characters")
    
    return generated_output

# ============================================================================
# Main Inference Functions
# ============================================================================
def inference_from_json_file(model, tokenizer, json_file_path: str, output_file_path: str = None):
    """
    Run inference on a JSON file containing input data
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        json_file_path: Path to JSON file with input data
        output_file_path: Optional path to save output (if None, prints to console)
    """
    
    print(f"Loading input from: {json_file_path}")
    
    # Load JSON input
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single object and array of objects
    if isinstance(data, list):
        inputs = data
    elif isinstance(data, dict) and 'input' in data:
        inputs = [data]
    else:
        inputs = [{"input": data}]
    
    results = []
    
    # Process each input
    for i, item in enumerate(inputs):
        print(f"\n{'='*80}")
        print(f"Processing input {i+1}/{len(inputs)}")
        print(f"{'='*80}")
        
        # Extract input data
        if 'input' in item:
            input_data = item['input']
        else:
            input_data = item
        
        # Generate output
        generated_output = generate_output(model, tokenizer, input_data)
        
        # Store result
        result = {
            "input": input_data,
            "generated_output": generated_output
        }
        results.append(result)
        
        # Print preview
        print(f"\nGenerated Output Preview (first 500 chars):")
        print("-" * 80)
        print(generated_output[:500])
        if len(generated_output) > 500:
            print("... (output truncated)")
        print("-" * 80)
    
    # Save or print results
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_file_path}")
    else:
        print("\n" + "="*80)
        print("COMPLETE OUTPUT")
        print("="*80)
        for i, result in enumerate(results):
            print(f"\n--- Output {i+1} ---")
            print(result['generated_output'])
            print()
    
    return results

def inference_from_dict(model, tokenizer, input_data: dict) -> str:
    """
    Run inference on a dictionary input
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        input_data: Dictionary with input specifications
    
    Returns:
        Generated Typst code
    """
    return generate_output(model, tokenizer, input_data)

# ============================================================================
# CLI Interface
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Typst code using fine-tuned Qwen model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file with input data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save generated output (default: print to console)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.LORA_ADAPTER_PATH,
        help=f"Path to LoRA adapter (default: {config.LORA_ADAPTER_PATH})"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=config.BASE_MODEL_NAME,
        help=f"Base model name (default: {config.BASE_MODEL_NAME})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.MAX_NEW_TOKENS,
        help=f"Maximum tokens to generate (default: {config.MAX_NEW_TOKENS})"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.TEMPERATURE,
        help=f"Generation temperature (default: {config.TEMPERATURE})"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM)"
    )
    
    args = parser.parse_args()
    
    # Update config
    config.LORA_ADAPTER_PATH = args.model
    config.BASE_MODEL_NAME = args.base_model
    config.MAX_NEW_TOKENS = args.max_tokens
    config.TEMPERATURE = args.temperature
    config.USE_4BIT = not args.no_4bit
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        config.BASE_MODEL_NAME,
        config.LORA_ADAPTER_PATH,
        config.USE_4BIT
    )
    
    # Run inference
    inference_from_json_file(model, tokenizer, args.input, args.output)

# ============================================================================
# Example Usage in Python
# ============================================================================
if __name__ == "__main__":
    # Check if running from command line
    if len(sys.argv) > 1:
        main()
    else:
        # Example usage in Python script
        print("EXAMPLE USAGE:")
        print("="*80)
        print()
        print("Command line usage:")
        print("  python inference_qwen_lora.py --input input.json --output output.json")
        print()
        print("Python script usage:")
        print("""
# Load model once
model, tokenizer = load_model_and_tokenizer(
    config.BASE_MODEL_NAME, 
    config.LORA_ADAPTER_PATH,
    use_4bit=True
)

# Example 1: From JSON file
inference_from_json_file(
    model, 
    tokenizer, 
    "input.json", 
    "output.json"
)

# Example 2: From dictionary
input_data = {
    "starting_page": 1,
    "academic_year": "2025-26",
    "department": "B.E/Dept of CSE/BNMIT",
    "project_title": "My Project Title",
    "contents_txt": "Your chapter content here..."
}
output = inference_from_dict(model, tokenizer, input_data)
print(output)

# Example 3: Batch processing multiple files
for input_file in ["input1.json", "input2.json", "input3.json"]:
    output_file = input_file.replace("input", "output")
    inference_from_json_file(model, tokenizer, input_file, output_file)
""")
        print("="*80)

"""
DETAILED USAGE EXAMPLES:
========================

1. COMMAND LINE USAGE:
   
   # Basic usage - print to console
   python inference_qwen_lora.py --input test_input.json
   
   # Save output to file
   python inference_qwen_lora.py --input test_input.json --output generated_output.json
   
   # Custom model path
   python inference_qwen_lora.py --input test_input.json --model ./my_custom_lora
   
   # Adjust generation parameters
   python inference_qwen_lora.py --input test_input.json --max-tokens 3000 --temperature 0.5
   
   # Use full precision (no quantization)
   python inference_qwen_lora.py --input test_input.json --no-4bit

2. PYTHON SCRIPT USAGE:

   from inference_qwen_lora import load_model_and_tokenizer, inference_from_dict
   
   # Load model once (reuse for multiple inferences)
   model, tokenizer = load_model_and_tokenizer(
       "Qwen/Qwen-1_8B",
       "./qwen-lora-finetuned",
       use_4bit=True
   )
   
   # Single inference
   input_data = {
       "starting_page": 10,
       "academic_year": "2025-26",
       "department": "B.E/Dept of CSE/BNMIT",
       "project_title": "Machine Learning Project",
       "contents_txt": "Chapter content goes here..."
   }
   
   output = inference_from_dict(model, tokenizer, input_data)
   print(output)
   
   # Save to file
   with open("output.typ", "w", encoding="utf-8") as f:
       f.write(output)

3. INPUT JSON FORMAT:

   Single input:
   {
     "input": {
       "starting_page": 1,
       "academic_year": "2025-26",
       "department": "B.E/Dept of CSE/BNMIT",
       "project_title": "Project Title",
       "contents_txt": "Your content..."
     }
   }
   
   Multiple inputs (batch):
   [
     {
       "input": { ... }
     },
     {
       "input": { ... }
     }
   ]

4. GENERATION PARAMETERS:
   
   --temperature (0.1-2.0):
     - 0.1-0.5: More deterministic, consistent output
     - 0.7-0.9: Balanced creativity
     - 1.0-2.0: More creative, varied output
   
   --max-tokens:
     - Controls maximum length of generated code
     - Adjust based on your document complexity
   
   --top-p (0.1-1.0):
     - Nucleus sampling threshold
     - Lower = more focused, Higher = more diverse

5. COLAB USAGE:

   # Upload this script and your model to Colab
   !python inference_qwen_lora.py --input /content/test_input.json --output /content/output.json
"""
