import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import os

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_story(prompt):
    input_text = f"Write a detailed and creative story about {prompt}."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.9,
        top_p=0.95,
        do_sample=True
    )

    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

def text_to_speech(text):
    engine = pyttsx3.init()

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    file_path = "outputs/story_audio.mp3"
    engine.save_to_file(text, file_path)
    engine.runAndWait()

    return file_path

def main():
    topic = input("Enter story topic: ")

    print("Generating story on GPU...\n")
    story = generate_story(topic)

    with open("outputs/story.txt", "w", encoding="utf-8") as f:
        f.write(story)

    print("Converting to voice...\n")
    audio_path = text_to_speech(story)

    print("Done ✅")
    print("Story saved at outputs/story.txt")
    print("Audio saved at", audio_path)

if __name__ == "__main__":
    main()