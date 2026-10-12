import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from data_pipeline import load_and_preprocess_data, MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE

# Training Hyperparameters
EMBEDDING_DIM = 64
EPOCHS = 150
BATCH_SIZE = 16

def build_intent_model(num_classes):
    """Compiles a deep sequence-to-intent mapping network."""
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Assumes 'intents.json' resides in the execution workspace
    try:
        X, y, num_classes = load_and_preprocess_data('intents.json')
        model = build_intent_model(num_classes)
        
        print("[INFO] Initiating model compiling and training loops...")
        model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        
        # Save compiled neural weights
        model.save('chatbot_intent_model.h5')
        print("[SUCCESS] Core model artifacts successfully saved to workspace.")
    except FileNotFoundError:
        print("[ERROR] Please provide a valid 'intents.json' file to begin model compilation.")
