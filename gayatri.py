# Import necessary libraries
import numpy as np
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download Microsoft's DialoGPT model and tokenizer
checkpoint = "microsoft/DialoGPT-medium"

# Download and cache tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Set pad_token to eos_token to avoid attention mask issues
tokenizer.pad_token = tokenizer.eos_token

# Download and cache pre-trained model
model = AutoModelForCausalLM.from_pretrained(checkpoint)


# A ChatBot classpython
class ChatBot():
    def __init__(self):
        """Initialize the chatbot"""
        self.chat_history_ids = None  # Store chat history for continuity
        self.bot_input_ids = None  # Input IDs for the bot
        self.end_chat = False  # Flag to indicate when to exit chat
        self.welcome()  # Display welcome message
        
    def welcome(self):
        """Display a welcome message when the chatbot starts"""
        print("Initializing ChatBot ...")
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        time.sleep(3)
        greeting = np.random.choice([
            "Welcome, I am ChatBot, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"
        ])
        print("ChatBot >>  " + greeting)
        
    def user_input(self):
        """Receive and process user input"""
        text = input("User    >> ")
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            self.end_chat = True  # End conversation
            print('ChatBot >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting ChatBot ...')
        else:
            # Encode the new user input, add eos_token, and convert to tensor
            self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    def bot_response(self):
        """Generate and return a response from the bot"""
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1)
        else:
            self.bot_input_ids = self.new_user_input_ids
        
        # Create an attention mask
        attention_mask = torch.ones(self.bot_input_ids.shape, dtype=torch.long)

        # Generate response while limiting total history to 1000 tokens
        self.chat_history_ids = model.generate(
            self.bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask  # Ensure proper attention handling
        )

        # Decode the last bot response
        response = tokenizer.decode(
            self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )

        if response == "":
            response = self.random_response()  # Use fallback response

        print('ChatBot >>  ' + response)

    def random_response(self):
        """Return a fallback response if the model generates an empty string"""
        return np.random.choice([
            "I'm not sure.", 
            "Can you rephrase that?", 
            "I don't know."
        ])


# Instantiate and start chatting
bot = ChatBot()

while True:
    bot.user_input()
    if bot.end_chat:
        break
    bot.bot_response()
