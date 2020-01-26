import sacred
import h5py
import cv2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import probabilistic_drnn_model
import visualization
import tf_helpers as tfh

HD = h5py.File('../../DQN/car_racing.hdf5', 'r+')

MIN_REWARD = HD['rewards'][()].min()
MAX_REWARD = HD['rewards'][()].max()
MIN_REWARD, MAX_REWARD

CHECKPOINTS = HD['checkpoints'][()]
OBSERVATIONS = HD['observations']
REWARDS = (HD['rewards'] - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
ACTIONS = HD['actions'][()]

EX = sacred.Experiment(name='DRNN Car Racing 3')


def data_input(W, H, SEQ_LEN):
    for c, a, o, r in zip(CHECKPOINTS, ACTIONS, OBSERVATIONS, REWARDS):
        c = c[:SEQ_LEN]
        a = a[:SEQ_LEN]
        o = o[:SEQ_LEN] / 255.0
        r = r[:SEQ_LEN]

        yield (c, a), (o, r)


def on_batch_begin(model, input_generator):
    def sampler():
        X, Y = list(input_generator.take(1))[0]
        return X, Y[0]

    visualization.plot_pairwise_frames(
        sampler=sampler,
        hypotheses=lambda x: model.net.predict(x)[0][0],
    )

    if 'loss' in model.net.history.history:
        loss = model.net.history.history['loss'][-1]
        val_loss = model.net.history.history['val_loss'][-1]
        EX.log_scalar('loss', loss)
        EX.log_scalar('val_loss', val_loss)

    pred_rollout_name = 'pred_rollout_2.png'
    plt.savefig(pred_rollout_name)
    EX.add_artifact(pred_rollout_name)
    plt.show()


MODEL = None


@EX.main
def main(SEQ_LEN, W, H, internal_size, batch_size, steps_per_epoch, lr,
         weight_decay, should_preload_model):
    global MODEL
    input_generator = tfh.make_dataset(
        lambda: data_input(W, H, SEQ_LEN),
        batch_size,
    )

    #### Model
    MODEL = probabilistic_drnn_model.DRNN(
        internal_size=internal_size,
        lr=lr,
        weight_decay=weight_decay,
    )
    MODEL.net.summary()

    model_img_name = 'conditioned_drnn_64_scaled_renderer.png'
    tf.keras.utils.plot_model(
        MODEL.net,
        show_layer_names=True,
        to_file=model_img_name,
        show_shapes=True,
        expand_nested=True,
        rankdir='TB',
        dpi=90,
    )
    EX.add_artifact(model_img_name)

    #### Callbacks
    callbacks = [
        tfh.model_persistor(
            MODEL,
            should_preload_model=should_preload_model,
            cp_file_name='cp-drnn-{epoch:0004d}-{loss:.5f}.hdf5',
        ),
        tfh.on_batch_begin(lambda: on_batch_begin(MODEL, input_generator)),
        MODEL.tb_callback,
    ]

    #### Training
    MODEL.net.fit_generator(
        generator=input_generator,
        validation_data=input_generator,
        validation_steps=2,
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    EX.add_config(
        SEQ_LEN = 128,
        W = 128,
        H = 128,
        internal_size = 32,
        batch_size = 4,
        steps_per_epoch = 128,
        lr = 0.001,
        weight_decay = 0.0001,
        should_preload_model = True,
    )
    EX.run()
