
Some sample training models
#    model v1 - Conv2d
    model = Sequential()
    model.add(Conv2D(64, (1, 2), strides=(1,2), activation='relu', input_shape=(102, 1000, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (1, 2), strides=(1,2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(TimeDistributed(LSTM(32, activation='tanh')))
    model.add(Flatten())
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights('model-5-5-4.h5')

#    model v2 - mlp
    model = Sequential()
    model.add(Dense(units=512, kernel_initializer='uniform', activation='relu', input_shape=(1000, 102)))
    model.add(Dense(units=32, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

 #   model v3 -- using conv1d
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, padding='valid', strides=3, input_shape=(1000, 102), kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv1D(filters=32, kernel_size=5, padding='valid', strides=2, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Conv1D(filters=16, kernel_size=5, padding='valid', strides=2, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Flatten())
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
    model.load_weights('model-3.h5')

#    model v4 - rnn
    model = Sequential()
    model.add(Bidirectional(CuDNNGRU(units=64, kernel_regularizer=regularizers.l2(0.001), return_sequences=True, input_shape=(1000, 102))))
    model.add(Bidirectional(CuDNNGRU(units=32, kernel_regularizer=regularizers.l2(0.001), return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=64, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=16, kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))


