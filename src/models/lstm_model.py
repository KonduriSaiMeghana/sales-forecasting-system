import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_length=30):

    X = []
    y = []

    for i in range(len(data) - seq_length):

        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)


def train_lstm(train, test):

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(
        train[['sales']]
    )

    test_scaled = scaler.transform(
        test[['sales']]
    )

    X_train, y_train = create_sequences(train_scaled)

    X_test, y_test = create_sequences(test_scaled)

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32
    )

    preds = model.predict(X_test)

    preds = scaler.inverse_transform(preds)

    return model, preds