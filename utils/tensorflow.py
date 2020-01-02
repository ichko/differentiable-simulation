import os
import tensorflow as tf


def on_batch_begin(func):
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, batch, logs=None):
            func()

    return CustomCallback()


def model_persistor(model):
    checkpoint_dir = '.checkpoints/'
    cp_file_name = 'cp.{epoch:0004d}-{loss:.5f}.hdf5'

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    latest = tf.train.latest_checkpoint(checkpoint_dir,
                                        latest_filename=cp_file_name)
    sorted_by_date = sorted(
        [checkpoint_dir + f for f in os.listdir(checkpoint_dir)],
        key=os.path.getmtime)

    if latest is None and len(sorted_by_date) > 0:
        latest = sorted_by_date[-1]

    if latest:
        try:
            model.net.load_weights(latest)
        except Exception as e:
            print('Error on loading persisted model: %s' % e)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + cp_file_name,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        save_freq='epoch')

    return checkpoint_callback
