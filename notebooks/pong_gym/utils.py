import matplotlib.pyplot as plt
import tensorflow as tf


def plot_pairwise_frames(dataset, predictor, which_frames):
    X, Y = list(dataset.take(1))[0]
    (input,), (frames, _reward, _done) = X, Y
    pred_frames, _pred_reward, _pred_done = predictor([input])

    plot_size = 2
    num_images = len(which_frames)
    fig, axs = plt.subplots(2, num_images, figsize=(
        plot_size * num_images, plot_size * 2))

    for i, f in enumerate(which_frames):
        l, r = (axs[0, i], axs[1, i])

        l.imshow(frames[0, f], cmap='bwr')
        r.imshow(pred_frames[0, f], cmap='bwr')

        l.set_xticklabels([])
        r.set_xticklabels([])
        l.set_yticklabels([])
        r.set_yticklabels([])

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
