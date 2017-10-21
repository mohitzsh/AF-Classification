import csv
import random
import numpy as np

def dataset_count(T, training_percent):
    training_count = (int)(training_percent * T)
    validation_count = T - training_count
    return training_count, validation_count

def reservoir_sampling(indices, training_count, validation_count):
    training_samples = random.sample(indices, training_count)
    validation_samples = [index for index in indices if index not in training_samples]
    return training_samples, validation_samples

def get_label(label):
    label = label.strip()
    if label == 'N':
        return '0'
    elif label == 'A':
        return '1'
    elif label == 'O':
        return '2'
    else:
        return '3'

def create_dataset(samples, labels, test_labels, training_samples, validation_samples):
    writer = open('training_set.csv', 'w')
    for index in training_samples:
        writer.write(samples[index] + '\n')
    writer.close()
    writer = open('training_labels.txt', 'w')
    for index in training_samples:
        writer.write(get_label(labels[index]) + '\n')
    writer.close()
    writer = open('validation_set.csv', 'w')
    for index in validation_samples:
        writer.write(samples[index] + '\n')
    writer.close()
    writer = open('validation_labels.txt', 'w')
    for index in validation_samples:
        writer.write(get_label(labels[index]) + '\n')
    writer.close()
    writer = open('test_labels.txt', 'w')
    for label in test_labels:
        writer.write(get_label(label) + '\n')
    writer.close()

if __name__ == "__main__":  
    training_count, validation_count = dataset_count(7000, 0.8)
    samples = []
    reader = csv.reader(open('features.csv', 'r'))
    for row in reader:
        samples.append(','.join(row))
    training_samples, validation_samples = reservoir_sampling(range(7000), training_count, validation_count)
    with open('training/reference_train_test.txt') as f:
        training_labels = f.readlines()
    with open('testing/reference_test.txt') as f:
        test_labels = f.readlines()
    create_dataset(samples, training_labels, test_labels, training_samples, validation_samples)