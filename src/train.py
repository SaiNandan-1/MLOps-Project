import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os

def main():
    print("Training model...")
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
    x_train_pad = pad_sequences(x_train, maxlen=500)
    y_train_one_hot = to_categorical(y_train, num_classes=46)

    model = Sequential([
        Embedding(10000, 128, input_length=500),
        Bidirectional(LSTM(64)),
        Dense(46, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train_pad, y_train_one_hot, epochs=1, batch_size=128) # Kept to 1 epoch for testing speed

    os.makedirs("models", exist_ok=True)
    model.save("models/reuters_bilstm.keras")
    print("Model saved.")

if __name__ == "__main__":
    main()