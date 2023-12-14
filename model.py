from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import os
import pickle

max_fatures = 2000
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


model = Sequential()
model.add(Embedding(2000, 128, input_length=47))
model.add(LSTM(196, dropout=0.2,return_sequences=True))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, input_dim=47, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.load_weights('my_model_weights.h5')