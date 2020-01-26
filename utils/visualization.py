import matplotlib.pyplot as plt
import numpy as np
import cv2


def to_video(feed, name, size=1):
    def normalize(video):
        min = np.min(video)
        max = np.max(video)
        np_version = np.array(((video - min) / (max - min) * 255.0))
        return np_version.astype(np.uint8)

    _, W, H, _ = feed[0].shape
    feed = [normalize(v) for v in feed]
    split_frame = np.concatenate(feed, axis=2)

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(name, fourcc, 30, (W * len(feed), H))

    for frame in split_frame:
        f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(f)

    video.release()


def plot_pairwise_frames(sampler, hypotheses, num_samples=10):
    X, Y = sampler()
    Y = Y[0]
    pred_Y = hypotheses(X)
    rollout_size = len(pred_Y)

    plot_size = 2
    fig, axs = plt.subplots(
        2,
        num_samples,
        figsize=(plot_size * num_samples, plot_size * 2),
    )

    plot_range = range(
        1,
        rollout_size - num_samples,
        rollout_size // num_samples,
    )

    for i, f in enumerate(plot_range):
        l, r = (axs[0, i], axs[1, i])

        l.imshow(Y[f], cmap='bwr')
        r.imshow(pred_Y[f], cmap='bwr')

        l.set_xticklabels([])
        r.set_xticklabels([])
        l.set_yticklabels([])
        r.set_yticklabels([])

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)