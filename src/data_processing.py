import os
import tensorflow as tf


def get_class_names(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return sorted(classes)


def process_path(file_path, class_names, img_size=(64, 64)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, img_size)
    img = img / 255.0

    parts = tf.strings.split(file_path, os.path.sep)
    class_name = parts[-2]
    label = tf.argmax(class_name == class_names)
    return img, label


def load_dataset(data_dir, batch_size=64):
    class_names = get_class_names(data_dir)
    file_pattern = os.path.join(data_dir, '*/*.jpeg')
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    def map_func(path):
        return process_path(path, class_names)

    dataset = files.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset, class_names