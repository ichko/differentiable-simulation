import os
import pickle
import numpy as np
import cv2
from argparse import Namespace
import torch
from config import hparams


def persist(func, path, override=False):
    if not override and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    result = func()
    with open(path, 'wb+') as f:
        pickle.dump(result, f)

    return result


def produce_video(vid_path, model, env, hparams):
    rollout = play_model(env, model, num_episodes=5)
    frames_to_video(vid_path, rollout, fps=10, frame_size=None)


def frames_to_video(name, frames, fps=30, frame_size=None):
    frames_iterator = iter(frames)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    frame = next(frames_iterator)
    H, W = frame.shape[:2]
    frame_size = (W, H) if frame_size is None else frame_size
    video = cv2.VideoWriter(name, fourcc, fps, frame_size)

    for frame in frames_iterator:
        frame = cv2.resize(frame, frame_size, cv2.INTER_AREA)
        if frame.max() <= 1:
            frame *= 255
        frame = frame.astype(np.uint8)

        video.write(frame)

    video.release()


def render_env(env, hparams):
    frame = env.render('rgb_array')
    frame = cv2.resize(frame, hparams.frame_size)
    return frame


def play_model(env, model, agent=None, num_episodes=5):
    model = model.to('cpu')

    with torch.no_grad():
        for i in range(num_episodes):
            def get_action():
                return i if i < env.action_space.n else env.action_space.sample()

            agent = get_action if agent is None else agent

            env.reset()

            first = render_env(env, hparams)

            action = agent()

            # env.step(action)
            # second = render_env(env, hparams)

            # preconditions = np.concatenate([first, second], axis=-1)
            preconditions = np.concatenate([first], axis=-1)
            preconditions = np.transpose(preconditions, (2, 0, 1))

            # model.eval()
            model.reset(precondition=preconditions)

            done = False
            while not done:
                action = agent()
                _, _, done, _ = env.step(action)
                pred_frame, _, _, _ = model.step(action)
                pred_frame = np.transpose(pred_frame, (1, 2, 0))
                frame = render_env(env, hparams)

                diff = abs(
                    pred_frame.astype(np.float32) - frame
                ).astype(np.uint8)

                screen = np.concatenate([frame, pred_frame, diff], axis=1)

                screen = cv2.resize(
                    screen,
                    (int(hparams.frame_size[0] * 3 * 2.5),
                     int(hparams.frame_size[1] * 2.5)),
                )

                screen = cv2.putText(
                    screen,
                    f'ep:{i + 1}', (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1,
                    cv2.LINE_AA,
                )

                yield screen

    model = model.to('cuda')
