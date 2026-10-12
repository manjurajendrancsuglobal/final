import json
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration Parameters
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 20
PADDING_TYPE = 'post'
TRUNCATING_TYPE = 'post'

def load_and_preprocess_data(dataset_path):
    """Loads closed-domain intents, extracts patterns, and maps them to responses."""
    with open(dataset_path, 'r') as file:
        data = json.load(file)
        
    training_sentences = []
    training_labels = []
    labels_to_responses = {}
    
    # Extract data structures
    for idx, intent in enumerate(data['intents']):
        tag = intent['tag']
        labels_to_responses[idx] = intent['responses']
        
        for pattern in intent['patterns']:
            training_sentences.append(pattern.lower())
            training_labels.append(idx)
            
    # Tokenization and Text-to-Sequence Mapping
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)
    
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, 
                                     padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)
    
    # Save artifacts for runtime deployment
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('response_mappings.pickle', 'wb') as handle:
        pickle.dump(labels_to_responses, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return np.array(padded_sequences), np.array(training_labels), len(data['intents'])

if __name__ == "__main__":
    # Example structure for execution validation
    print("[INFO] Initializing training data pipeline processing...")
