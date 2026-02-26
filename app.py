import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import os

# =====================================================
# DEVICE CHECK
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =====================================================
# PROJECT PATHS
# =====================================================
BASE_DIR = r"L:\Projects\AI\gpu_story_ai"
CACHE_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =====================================================
# LOAD TOKENIZER & MODEL
# =====================================================
print("Loading model... (first time may download)")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=CACHE_DIR
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True
)

model.to(device)
model.eval()

# =====================================================
# STORY GENERATION FUNCTION
# =====================================================
def generate_story(prompt):

    messages = [
        {"role": "system", "content": "You are a creative and professional storyteller."},
        {"role": "user", "content": prompt}
    ]

    # Step 1: Apply chat template properly
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

    # Step 2: Tokenize into tensor
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    # Remove prompt part from output
    generated_tokens = output[:, inputs["input_ids"].shape[-1]:]

    story = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return story


# =====================================================
# TEXT TO SPEECH
# =====================================================
def text_to_speech(text):
    engine = pyttsx3.init()

    file_path = os.path.join(OUTPUT_DIR, "story_audio.mp3")

    engine.save_to_file(text, file_path)
    engine.runAndWait()

    return file_path


# =====================================================
# MAIN
# =====================================================
def main():
    topic = input("\nEnter story topic:\n\n")

    print("\nGenerating story...\n")
    story = generate_story(topic)

    story_path = os.path.join(OUTPUT_DIR, "story.txt")

    with open(story_path, "w", encoding="utf-8") as f:
        f.write(story)

    print("\nConverting to voice...\n")
    audio_path = text_to_speech(story)

    print("\nDone ✅")
    print("Story saved at:", story_path)
    print("Audio saved at:", audio_path)


if __name__ == "__main__":
    main()