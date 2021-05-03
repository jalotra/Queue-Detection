from keras import models
from utils import mean_square_error, mean_absolute_error

#load the trained model
model = models.load_model('./ShanghaiTech/part_B/weights/mcnn_val.hdf5', \
    custom_objects={'mean_absolute_error': mean_absolute_error, \
         'mean_square_error': mean_square_error })




if __name__ == "__main__":
    absolute_error = []
    squared_error = []
    # specifying the number of test to run
    num_test = 50
    for i in range(testing_img.shape[0])[:num_test]:
        inputs = np.reshape(testing_img[i], [1, *testing_img[i].shape[:2], 1])
        outputs = np.squeeze(model.predict(inputs))
        density_map = np.squeeze(test_labels[i])
        count = np.sum(density_map)
        prediction = np.sum(outputs)
        fg, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
        # plotting the density maps along with predicted count
        plt.suptitle(' '.join([
            'count:', str(round(count, 2)),
            'prediction:', str(round(prediction, 2))
        ]))
        ax0.imshow(np.squeeze(inputs))
        ax1.imshow(density_map * (255 / (np.max(density_map) - np.min(density_map))))
        plt.show()
        absolute_error.append(abs(count -  prediction))
        square_error.append((count -  prediction) ** 2)
    mean_absolute_error = np.mean(absolute_error)
    mean_square_error = np.mean(square_error)
    print('mean_absolute_error:', mean_absolute_error, 'mean_square_error:', mean_square_error)