import numpy as np
import cv2


def generate_data(env, agent, dataset_size, frame_size, precondition_size):
    print('> Generating data...')

    total_step = 0

    actions = np.zeros((dataset_size, 1), dtype=np.uint8)
    preconditions = np.zeros(
        (dataset_size, precondition_size * 3, *frame_size[::-1]),
        dtype=np.uint8)
    futures = np.zeros((dataset_size, 3, *frame_size[::-1]), dtype=np.uint8)

    while True:
        env.reset()
        done = False

        frames_queue = np.zeros(
            (precondition_size + 1, 3, *frame_size[::-1]),
            dtype=np.uint8,
        )

        episode_step = 0

        while not done:
            action = agent(env)
            _, _, done, _ = env.step(action)
            frame = env.render('rgb_array')
            frame = cv2.resize(frame, frame_size)
            frame = np.transpose(frame, (2, 0, 1))
            frame = frame.astype(np.uint8)

            frames_queue = np.roll(frames_queue, shift=-1, axis=0)
            frames_queue[-1] = frame

            episode_step += 1
            if episode_step >= precondition_size + 1:
                precondition = frames_queue[:precondition_size]
                future = frames_queue[-1]
                last_action = action

                actions[total_step] = last_action
                preconditions[total_step] = precondition.reshape(
                    precondition_size * 3,
                    *frame_size[::-1],
                )
                futures[total_step] = future

                total_step += 1
                if total_step >= dataset_size:
                    return actions, preconditions, futures
