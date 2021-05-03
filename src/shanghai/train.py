'''
 Implements the training schedule for the MultiColumnCNN defined in ./model.py
'''
from utils import x_y_generator
from keras import backend as K
from keras.callback import ModelCheckpoint
from utils import mean_absolute_error, mean_square_error


if __name__ == "__main__":
    # read training data
    train_generator = x_y_generator(train_paths, train_labels, batch_size=len(train_paths))
    training_img, train_labels = train_generator.__next__()
    
    # read validation data
    validation_generator = x_y_generator(validation_paths, validation_labels, batch_size=len(validation_paths))
    validating_img, validation_labels = validation_generator.__next__()
    
    # read test data
    test_generator = x_y_generator(test_paths, test_labels, batch_size=len(test_paths))
    testing_img, test_labels = test_generator.__next__()


    best_validation = ModelCheckpoint(
    filepath= 'mcnn_val.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'
)
    best_training = ModelCheckpoint(
        filepath= 'mcnn_train.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min'
    )
    
    input_shape = (None, None, 1)
    model = Multi_Column_CNN(input_shape)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[mean_absolute_error, mean_square_error])
    history = model.fit(
        x=training_img, y=train_labels, batch_size=1, epochs=100,
        validation_data=(validating_img, validation_labels),
        callbacks=[best_validation, best_training]
    )

    # Plot Loss over epochs and used earlyStopping 
    val_loss, loss = history.history['val_loss'], history.history['loss']
    loss = np.asarray(loss)
    plt.plot(loss, 'b')
    plt.legend(['loss'])
    plt.show()
    plt.plot(val_loss, 'r')
    plt.legend(['val_loss'])
    plt.show()
