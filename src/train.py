import os
import json
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense

def main():
    print("Training model...")
    
    # 1. Load and preprocess the data
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
    x_train_pad = pad_sequences(x_train, maxlen=500)
    y_train_one_hot = to_categorical(y_train, num_classes=46)

    # 2. Build the BiLSTM model
    model = Sequential([
        Embedding(10000, 128, input_length=500),
        Bidirectional(LSTM(64)),
        Dense(46, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train the model (saved to 'history' to extract metrics)
    history = model.fit(x_train_pad, y_train_one_hot, epochs=1, batch_size=128)

    # 4. Extract the exact numbers
    accuracy = history.history['accuracy'][-1]
    loss = history.history['loss'][-1]

    metrics = {
        "accuracy": float(accuracy),
        "loss": float(loss)
    }

    # 5. Save the scores to metrics.json for Streamlit to read
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
        
    print("Metrics saved to metrics.json")

    # 6. Save the actual model file
    os.makedirs("models", exist_ok=True)
    model.save("models/reuters_bilstm.keras")
    print("Model saved.")

if __name__ == "__main__":
    main()