import numpy as np
from utils import get_density_map 
import os, random, math, cv2, csv
from scipy.io import loadmat

#Makes useful directiories
def make_directories():
    input_images_path = ''.join(['./ShanghaiTech/part_B/train_data/images/'])
    output_path = './ShanghaiTech/processed_trainval/'
    training_images_path = ''.join((output_path, '/training_images/'))
    training_densities_path = ''.join((output_path, '/training_densities/'))
    validation_images_path = ''.join((output_path, '/validation_images/'))
    validation_densities_path = ''.join((output_path, '/valalidation_densities/'))
    
    ground_truth_path = ''.join(['./ShanghaiTech/part_B/train_data/ground-truth/'])
 
for dir in [output_path, training_images_path, training_densities_path, validation_images_path, validation_densities_path]:
    if not os.path.exists(dir):
    	os.makedirs(dir)

# Process both the training and testing data the same way  
def process_training_data():

    # Names
    input_images_path = ''.join(['./ShanghaiTech/part_B/train_data/images/'])
    output_path = './ShanghaiTech/processed_trainval/'
    training_images_path = ''.join((output_path, '/training_images/'))
    training_densities_path = ''.join((output_path, '/training_densities/'))
    validation_images_path = ''.join((output_path, '/validation_images/'))
    validation_densities_path = ''.join((output_path, '/valalidation_densities/'))
    
    ground_truth_path = ''.join(['./ShanghaiTech/part_B/train_data/ground-truth/'])

    seed = 95461354
    random.seed(seed)
    n = 400
    val_test_num = math.ceil(n*0.1)
    indices = list(range(1, n+1))
    random.shuffle(indices)
    
    for idx in range(1, n+1):
        i = indices[idx-1]
        image_info = loadmat(''.join((ground_truth_path, 'GT_IMG_', str(i), '.mat')))['image_info']
        input_image = ''.join((input_images_path, 'IMG_',str(i), '.jpg'))
        img = cv2.imread(input_image, 0)
        height, width = img.shape
        new_width, new_height = width / 8, height / 8
        new_width, new_height = int(new_width / 8) * 8, int(new_height / 8) * 8
        annotation_Points =  image_info[0][0][0][0][0] - 1
        if width <= new_width * 2:
            img = cv2.resize(img, [h, new_width*2+1], interpolation=cv2.INTER_LANCZOS4)
            annotation_Points[:, 0] = annotation_Points[:, 0] * 2 * new_width / width
        if height <= new_height * 2:
            img = cv2.resize(img, [new_height*2+1, w], interpolation=cv2.INTER_LANCZOS4)
            annotation_Points[:, 1] = annotation_Points[:,1] * 2 * new_height / height
        height, width = img.shape
        x_width, y_width = new_width + 1, width - new_width
        x_height, y_height = new_height + 1, height - new_height
    
        image_density = get_density_map(img, annotation_Points)
        for j in range(1, 10):
    
            x = math.floor((y_width - x_width) * random.random() + x_width)
            y = math.floor((y_height - x_height) * random.random() + x_height)
            x1, y1 = x - new_width, y - new_height
            x2, y2 = x + new_width - 1, y + new_height - 1
            base_image = im[y1-1:y2, x1-1:x2]
            base_image_density = image_density[y1-1:y2, x1-1:x2]
            base_image_annPoints = annotation_Points[
                list(
                    set(np.where(np.squeeze(annotation_Points[:,0]) > x1)[0].tolist()) &
                    set(np.where(np.squeeze(annotation_Points[:,0]) < x2)[0].tolist()) &
                    set(np.where(np.squeeze(annotation_Points[:,1]) > y1)[0].tolist()) &
                    set(np.where(np.squeeze(annotation_Points[:,1]) < y2)[0].tolist())
                )
            ]
    
            base_image_annPoints[:, 0] = base_image_annPoints[:, 0] - x1
            base_image_annPoints[:, 1] = base_image_annPoints[:, 1] - y1
            img_idx = ''.join((str(i), '_',str(j)))
    
            if idx < val_test_num:
                cv2.imwrite(''.join([validation_images_path, img_idx, '.jpg']), base_image)
                with open(''.join([validation_densities_path, img_idx, '.csv']), 'w', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerows(base_image_density)
            else:
                cv2.imwrite(''.join([training_images_path, img_idx, '.jpg']), base_image)
                with open(''.join([training_densities_path, img_idx, '.csv']), 'w', newline='') as output:
                    writer = csv.writer(output)
                    writer.writerows(base_image_density)
    print("Training Files processed successfully!")


def process_testing_data():
    images_path = ''.join(['./ShanghaiTech/part_B/test_data/images/'])
    ground_truth_path = ''.join(['./ShanghaiTech/part_B/test_data/ground-truth/'])
    ground_truth_csv = ''.join(['./ShanghaiTech/part_B/test_data/ground-truth_csv/'])
    
    n = 316
    
    for i in range(1, n+1):
        image_info = loadmat(''.join((ground_truth_path, 'GT_IMG_', str(i), '.mat')))['image_info']
        input_img  = ''.join((images_path, 'IMG_', str(i), '.jpg'))
        img = cv2.imread(input_img, 0)
        annotationPoints =  image_info[0][0][0][0][0] - 1
        image_density = get_density_map(img, annotationPoints)
        with open(''.join([ground_truth_csv, 'IMG_', str(i), '.csv']), 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerows(image_density)
    print("Testing data processed successfully!")




if __name__ == "__main__":
    make_directories()
    process_training_data()
    process_testing_data()