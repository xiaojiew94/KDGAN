




print(end_point_t.shape, end_point_t.dtype)


def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)


def create_tfrecord(infile, is_training=False):


def main(_):
    # create_tfrecord(config.valid_file, is_training=False)
    # create_tfrecord(config.train_file, is_training=True)
    
    labels = utils.load_collection(config.label_file)
    # print(labels)

    imagenet_labels = set()
    label_names = imagenet.create_readable_names_for_imagenet_labels()
    # for label_id, names in label_names.items():
    #     print('{}: {}'.format(label_id, names))
    for label in labels:
        label_ids = []
        for label_id, names in label_names.items():
            if label in names:
                label_ids.append(label_id)
        print('{}: {}'.format(label, label_ids))


if __name__ == '__main__':
    tf.app.run()