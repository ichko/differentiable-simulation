from game_of_life import random_state, next_frame
from renderer import Renderer

if __name__ == '__main__':
  W, H = 30, 30
  game = random_state(W, H)
  renderer = Renderer(500, 500, 'Conway')

  cell_size = 10
  while renderer.is_running:
    renderer.update()

    for x in range(W):
      for y in range(H):
        if game[x, y]:
          pos_x = x - W / 2
          pos_y = y - H / 2

          renderer.rect(
            pos_x * cell_size,
            pos_y * cell_size,
            cell_size,
            cell_size,
            fill='#fff'
          )

    game = next_frame(game)

