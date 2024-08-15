import torch
from transformers import pipeline, AutoTokenizer
import argparse

def load_model(model_name):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipe

def chat_with_model(pipe):
    conversation = []
    print("Chat started. Type 'quit' to exit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        conversation.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        
        response = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7)
        model_response = response[0]['generated_text'].split(prompt)[-1].strip()
        
        print(f"Model: {model_response}")
        conversation.append({"role": "assistant", "content": model_response})

def main():
    parser = argparse.ArgumentParser(description="Chat with a Hugging Face model")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Name of the Hugging Face model to use")
    args = parser.parse_args()

    pipe = load_model(args.model)
    chat_with_model(pipe)

if __name__ == "__main__":
    main()