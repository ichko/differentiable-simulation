import os
import tensorflow as tf
import tensorflow.keras.layers as kl


def drnn(*args, **kwargs):
    return DRNNLayer(*args, **kwargs)


class DRNNLayer(kl.Layer):
    def __init__(self, type, size, skip, name, stateful=False):
        super(DRNNLayer, self).__init__()

        types = {'gru': kl.GRU, 'lstm': kl.LSTM}
        assert type in types

        self.skip_size = skip

        self.explode = kl.Lambda(
            lambda x: tf.space_to_batch(x, [skip], paddings=[[0, 0]]),
            name='space_to_batch_%s' % name,
        )

        self.rnn = types[type](
            size,
            activation='tanh',
            return_sequences=True,
            stateful=stateful,
            name='rnn_%s_%s' % (type, name),
        )

        self.implode = kl.Lambda(
            lambda x: tf.batch_to_space(x, [skip], crops=[[0, 0]]),
            name='batch_to_space_%s' % name,
        )

    def call(self, x, initial_state=None):
        if self.skip_size > 1:
            x = self.explode(x)
        x = self.rnn(x, initial_state=initial_state)
        if self.skip_size > 1:
            x = self.implode(x)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'skip_size': self.skip_size,
            'explode': self.explode,
            'rnn': self.rnn,
            'implode': self.implode,
        })

        return config


def tf_print(func=lambda x: x):
    def printer(x):
        mapped_x = func(x)
        tf.print(mapped_x)
        return x

    return kl.Lambda(printer, name='printer')


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

    latest = tf.train.latest_checkpoint(
        checkpoint_dir,
        latest_filename=cp_file_name,
    )
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


def get_generator_shape(generator):
    # the output shapes is required because of
    # https://github.com/tensorflow/tensorflow/issues/24520
    def map_to_shape(el):
        return tuple(map_to_shape(e) for e in el) \
            if type(el) is tuple \
            else el.shape
        
    return map_to_shape(next(generator()))


def get_generator_type(generator):
    def map_to_type(el):
        return tuple(map_to_type(e) for e in el) \
            if type(el) is tuple \
            else el.dtype
        
    return map_to_type(next(generator()))


def make_dataset(generator, batch_size=1):
    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=get_generator_type(generator),
        output_shapes=get_generator_shape(generator),
    )

    ds = ds.repeat()
    ds = ds.batch(batch_size)
    # ds = ds.map(lambda x,y : (x, y), num_parallel_calls=16)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds