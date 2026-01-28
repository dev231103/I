import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pretrained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def chat():
    chat_history_ids = None
    print("Chatbot: Hi! I'm a chatbot. Type 'quit' to exit.")

    while True:
        user_input = input("User: ")

        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break

        # Encode user input
        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token,
            return_tensors='pt'
        )

        # Append new user input to history
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Add attention mask
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.9,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Decode response
        bot_response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        print(f"Chatbot: {bot_response}")

if __name__ == "__main__":
    chat()
