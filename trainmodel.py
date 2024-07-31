import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore
from keras.callbacks import TensorBoard # type: ignore
from function import *

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

# Data preparation
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        sequence_valid = True  # Flag to check if the sequence is valid
        for frame_num in range(sequence_length):
            res_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            try:
                res = np.load(res_path, allow_pickle=True)
                if isinstance(res, np.ndarray) and len(res.shape) > 0 and res.shape[0] == 63:
                    window.append(res)
                else:
                    sequence_valid = False
                    break  # Exit loop if an invalid frame is found
            except Exception as e:
                sequence_valid = False
                break  # Exit loop if an error occurs

        if sequence_valid and len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        # Remove or comment out the following print statement to suppress skipping messages
        # elif not sequence_valid:
        #     print(f"Skipping sequence {sequence} for action {action} due to invalid frames or errors.")

X = np.array(sequences)
if len(labels) > 0:
    y = to_categorical(labels).astype(int)
else:
    raise ValueError("No labels available. Check the data processing.")

# Split data into training and validation sets (75% train, 25% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# TensorBoard callback
log_dir = os.path.join('Logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb_callback = TensorBoard(log_dir=log_dir)

# Model definition
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Model compilation
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Model training with validation
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, callbacks=[tb_callback])

# Plotting training & validation loss and accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
