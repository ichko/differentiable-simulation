from game_of_life import GameOfLife
from renderer import Renderer

if __name__ == '__main__':
  game = GameOfLife(50, 50)
  game.randomize()

  renderer = Renderer(500, 500, 'Conway')

  cell_size = 8
  while renderer.is_running:
    renderer.update()

    for x in range(game.width):
      for y in range(game.height):
        if game.is_alive(x, y):
          pos_x = x - game.width / 2
          pos_y = y - game.height / 2

          renderer.rect(
            pos_x * cell_size,
            pos_y * cell_size,
            cell_size,
            cell_size,
            fill='#fff'
          )

    game.next()

