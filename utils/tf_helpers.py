import os
import tensorflow as tf


def drnn_layer(type, size, skip_size, stateful, name):
    types = {
        'gru': tf.keras.layers.GRU,
        'lstm': tf.keras.layers.LSTM,
    }

    if type not in types:
        raise ValueError('type of cell should be in [gru, lstm]')

    explode = tf.keras.layers.Lambda(
        lambda x: tf.space_to_batch(x, [skip_size], paddings=[[0, 0]]),
        name='space_to_batch_%s' % name,
    )

    rnn = types[type](
        size,
        activation='tanh',
        return_sequences=True,
        stateful=stateful,
        name='rnn_%s_%s' % (type, name),
    )

    implode = tf.keras.layers.Lambda(
        lambda x: tf.batch_to_space(x, [skip_size], crops=[[0, 0]]),
        name='batch_to_space_%s' % name,
    )

    bn = tf.keras.layers.BatchNormalization(name='batch_norm_%s' % name)

    def call(x, initial_state=None):
        x = explode(x)
        x = rnn(x, initial_state=initial_state)
        x = bn(x)
        x = implode(x)
        return x

    return call


def tf_print(func=lambda x: x):
    def printer(x):
        mapped_x = func(x)
        tf.print(mapped_x)
        return x

    return tf.keras.layers.Lambda(printer, name='printer')


def on_batch_begin(func):
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, batch, logs=None):
            func()

    return CustomCallback()


def model_persistor(
    model,
    should_preload_model=True,
    checkpoint_dir='.checkpoints/',
    cp_file_name='cp.{epoch:0004d}-{loss:.5f}.hdf5',
):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    latest = tf.train.latest_checkpoint(checkpoint_dir,
                                        latest_filename=cp_file_name)
    sorted_by_date = sorted(
        [checkpoint_dir + f for f in os.listdir(checkpoint_dir)],
        key=os.path.getmtime,
    )

    if latest is None and len(sorted_by_date) > 0:
        latest = sorted_by_date[-1]

    if should_preload_model and latest:
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
        save_freq='epoch',
    )

    return checkpoint_callback
