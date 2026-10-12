import os
import sys
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Static Constraints
CONFIDENCE_THRESHOLD = 0.60
MAX_SEQUENCE_LENGTH = 20

def get_asset_path(relative_path):
    """Get the absolute path to a resource, dealing with PyInstaller temporary directories."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Resolve Dynamic Paths for Standalone Executable Environment
model_path = get_asset_path('chatbot_intent_model.h5')
tokenizer_path = get_asset_path('tokenizer.pickle')
response_path = get_asset_path('response_mappings.pickle')

# Load Deployment Assets Safely using Binary Read Modes
model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(response_path, 'rb') as handle:
    response_mappings = pickle.load(handle)

def clean_and_predict(user_input):
    """Processes interactive streams and screens out nonsensical outcomes."""
    sequence = tokenizer.texts_to_sequences([user_input.lower()])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    predictions = model.predict(padded, verbose=0)
    highest_intent_idx = np.argmax(predictions)
    confidence_score = predictions[highest_intent_idx]
    
    # Closed-Domain Fallback Guardrail
    if confidence_score < CONFIDENCE_THRESHOLD:
        return "I apologize, but that is outside my scope of knowledge. Could you please rephrase your question regarding our supported services?"
        
    matched_responses = response_mappings[highest_intent_idx]
    return random.choice(matched_responses)

def run_chat_session():
    """Launches the executable command line UI loop."""
    print("=================================================================")
    print("       NLP CLOSED-DOMAIN CHATBOT - WORKSPACE REPL TERMINAL       ")
    print("       Type 'quit' or 'exit' at any time to close connection.    ")
    print("=================================================================\n")
    
    while True:
        try:
            user_msg = input("You: ").strip()
            if user_msg.lower() in ['quit', 'exit']:
                print("Chatbot: Terminal session terminated. Goodbye!")
                break
            if not user_msg:
                continue
                
            bot_reply = clean_and_predict(user_msg)
            print(f"Chatbot: {bot_reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nChatbot: Connection interrupted. Goodbye!")
            break

if __name__ == "__main__":
    run_chat_session()
