import argparse
import numpy as np
import math
import struct
import matplotlib.pyplot as plt

def decode_image(image_path):
    # read the binary-based data
    bin_data = open(image_path, 'rb').read()

    # parse the content in the head (magic number、number of images、height、width)
    offset = 0
    fmt_header = '>iiii'   #'>iiii' means reading four unsigned int32 by using big endian
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)

    # parse dataset
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B' means reading 784 unsigned byte by using big endian
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images, image_size, num_images, num_rows, num_cols

def decode_label(label_path):
    # read the binary-based data
    bin_data = open(label_path, 'rb').read()

    # parse the content in the head (magic number、number of images)
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # parse dataset
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def print_result(p, ans):
    print("Posterior (in log scale):")
    for i in range(10):
        print(i, ": ", p[i])
    pred = np.argmin(p)
    print("Prediction: ", pred, ", Ans: ", ans, "\n")
    
    return 0 if pred == ans else 1

def print_image(image, row, col, mode):
    print("Imagination of numbers in Baysian classifier:\n")
    # discrete
    if mode == 0:
        for label in range(10):
            print(label, ":")
            for r in range(row):
                for c in range(col):
                    white = sum(image[label][r * row + c][:17])
                    black = sum(image[label][r * row + c][17:])
                    print(f'{1 if black > white else 0} ', end='')
                print()
            print()
    # continuous
    elif mode == 1:
        for label in range(10):
            print(label, ":")
            for r in range(row):
                for c in range(col):
                    print(f'{1 if image[label][r * row + c] > 128 else 0} ', end='')
                print()
            print()

def DiscreteMode(train_image_path, train_label_path, test_image_path, test_label_path):
    train_images, train_image_size, train_num_images, train_num_rows, train_num_cols = decode_image(train_image_path)
    train_labels = decode_label(train_label_path)
    
    prior = np.zeros((10), dtype=int)
    likelihood = np.zeros((10, train_image_size, 32), dtype=int)
    likelihood_sum = np.zeros((10, train_image_size), dtype=int)
    
    # training
    for count in range(train_num_images):
        label = int(train_labels[count])
        prior[label] += 1
        for pixel in range(train_image_size):
            pixel_value = int(train_images[count][pixel])
            likelihood[label][pixel][pixel_value // 8] += 1
    
    for i in range(10):
        for j in range(train_image_size):
            for k in range(32):
                likelihood_sum[i][j] += likelihood[i][j][k]
            
    test_images, test_image_size, test_num_images, test_num_rows, test_num_cols = decode_image(test_image_path)
    test_labels = decode_label(test_label_path)
    
    # testing
    error = 0
    for count in range(test_num_images):
        ans = int(test_labels[count])
        p = np.zeros((10), dtype=float)
        test_image = test_images[count]
        for label in range(10):
            p[label] += np.log(float(prior[label] / train_num_images))
            for pixel in range(test_image_size):
                temp = likelihood[label][pixel][int(test_image[pixel] / 8)]
                if temp == 0:
                    p[label] += np.log(float(1e-6 / likelihood_sum[label][pixel]))
                else:
                    p[label] += np.log(float(temp / likelihood_sum[label][pixel]))
        sumofp = sum(p)
        p /= sumofp
        error += print_result(p, ans)
    
    print_image(likelihood, test_num_rows, test_num_cols, 0)
    print("Error rate: ", float(error / test_num_images))

def ContinuousMode(train_image_path, train_label_path, test_image_path, test_label_path):
    train_images, train_image_size, train_num_images, train_num_rows, train_num_cols = decode_image(train_image_path)
    train_labels = decode_label(train_label_path)
    
    prior = np.zeros((10), dtype=float)
    var = np.zeros((10, train_image_size), dtype=float)
    mean = np.zeros((10, train_image_size), dtype=float)
    mean_square = np.zeros((10, train_image_size), dtype=float)
    
    # training
    for count in range(train_num_images):
        label = int(train_labels[count])
        prior[label] += 1
        for pixel in range(train_image_size):
            pixel_value = int(train_images[count][pixel])
            mean[label][pixel] += pixel_value
            mean_square[label][pixel] += (pixel_value ** 2)
    
    # Calculate mean and standard deviation
    for label in range(10):
        for pixel in range(train_image_size):
            mean[label][pixel] /= prior[label]
            mean_square[label][pixel] /= prior[label]
            var[label][pixel] = mean_square[label][pixel] - (mean[label][pixel] ** 2)
            var[label][pixel] = 1e-4 if var[label][pixel] == 0 else var[label][pixel]
    
    prior /= train_num_images
    prior = np.log(prior)
    
    test_images, test_image_size, test_num_images, test_num_rows, test_num_cols = decode_image(test_image_path)
    test_labels = decode_label(test_label_path)
    
    # testing
    error = 0
    for count in range(test_num_images):
        ans = int(test_labels[count])
        p = np.zeros((10), dtype=float)
        test_image = test_images[count]
        for label in range(10):
            p[label] += prior[label]
            for pixel in range(test_image_size):
                temp = np.log(1.0 / (np.sqrt(2.0 * np.pi * var[label][pixel]))) - ((test_image[pixel] - mean[label][pixel]) ** 2.0 / (2.0 * var[label][pixel]))
                p[label] += temp
        
        sumofp = sum(p)
        p /= sumofp
        error += print_result(p, ans)
    print_image(mean, test_num_rows, test_num_cols, 1)
    print("Error rate: ", float(error / test_num_images))

# default
train_image_path = 'train-images.idx3-ubyte'
train_label_path = 'train-labels.idx1-ubyte'
test_image_path = 't10k-images.idx3-ubyte'
test_label_path = 't10k-labels.idx1-ubyte'
mode = 1

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--TRAINING_IMAGE", type=str)
parser.add_argument("--TRAINING_LABEL", type=str)
parser.add_argument("--TESTING_IMAGE", type=str)
parser.add_argument("--TESTING_LABEL", type=str)
parser.add_argument("--MODE", type=int)
args = parser.parse_args()
train_image_path = args.TRAINING_IMAGE
train_label_path = args.TRAINING_LABEL
test_image_path = args.TESTING_IMAGE
test_label_path = args.TESTING_LABEL
mode = args.MODE

# naive baye classifier
if mode == 0:
    DiscreteMode(train_image_path, train_label_path, test_image_path, test_label_path)
elif mode == 1:
    ContinuousMode(train_image_path, train_label_path, test_image_path, test_label_path)

