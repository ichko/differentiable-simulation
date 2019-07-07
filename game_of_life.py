import numpy as np

class GameOfLife:
  def __init__(self, width, height):
    self.state = np.zeros([width, height])
    self.width = width
    self.height = height

  def randomize(self):
    self.state = np.random.randint(2, size=(self.width, self.height))

  def next(self):
    new_state = self.state.copy()
    height, width = self.state.shape
    for y in range(height):
      for x in range(width):
        arround = self.population_arround(x, y)
        if self.state[y, x] and (arround < 2 or arround > 3):
          new_state[y, x] = 0
        elif arround == 3:
          new_state[y, x] = 1

    self.state = new_state

  def population_arround(self, x, y):
    return sum([
      self.is_alive(x + dx, y + dy)
      for dx in [-1, 0, 1]
      for dy in [-1, 0, 1]
      if dx != 0 or dy != 0
    ])

  def is_alive(self, x, y):
    x = x % self.width
    y = y % self.height

    return self.state[y, x]

  def __repr__(self):
    return str(self.state)


if __name__ == '__main__':
  game = GameOfLife(10, 10)
  game.randomize()

  for _ in range(10):
    print(game)
    game.next()

