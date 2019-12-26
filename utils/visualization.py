import matplotlib.pyplot as plt


def plot_pairwise_frames(sampler, hypotheses, num_samples=10):
    X, Y = sampler()
    pred_Y = hypotheses(X)
    rollout_size = len(pred_Y)

    plot_size = 2
    fig, axs = plt.subplots(
        2, num_samples, figsize=(plot_size * num_samples, plot_size * 2)
    )

    for i, f in enumerate(range(1, rollout_size, rollout_size / num_samples)):
        l, r = (axs[0, i], axs[1, i])

        l.imshow(Y[0, f], cmap='bwr')
        r.imshow(pred_Y[0, f], cmap='bwr')

        l.set_xticklabels([])
        r.set_xticklabels([])
        l.set_yticklabels([])
        r.set_yticklabels([])

    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
