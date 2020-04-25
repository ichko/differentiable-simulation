from ple.games.flappybird import FlappyBird
from ple import PLE
import random

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    if p.game_over():  #check if the game is over
        p.reset_game()

    obs = p.getScreenRGB()
    action = random.sample(p.getActionSet(), 1)[0]
    reward = p.act(action)
    print(action, reward)