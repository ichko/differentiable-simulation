import math


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vec):
        self.x += vec.x
        self.y += vec.y

        return self

    def __repr__(self):
        return 'vec(%f, %f)' % (self.x, self.y)


def vec(x=0, y=0):
    return Vector(x, y)


def polar(angle, magnitude):
    return vec(
        math.cos(angle) * magnitude,
        math.sin(angle) * magnitude
    )


class Plank:
    def __init__(self, x):
        self.pos = vec(x, 0)
        self.width = 10
        self.height = 25
        self.speed = 2

    def render(self, renderer):
        renderer.rect(
            self.pos.x, self.pos.y,
            self.width, self.height,
            fill='#fff'
        )


class Ball:
    def __init__(self, x, y):
        self.pos = vec(x, y)
        self.direction = math.pi / 4
        self.speed = 2
        self.size = 10

    def render(self, renderer):
        renderer.arc(self.pos.x, self.pos.y, self.size, fill='#fff')


class PONG:
    def __init__(self):
        self.width = 100
        self.height = 100

        self.left_plank = Plank(5)
        self.right_plank = Plank(self.width - 15)
        self.ball = Ball(self.width / 2, self.height / 2)

    def update_plank(self, plank, inpt):
        plank.pos.y += inpt * plank.speed

        if plank.pos.y < 0:
            plank.pos.y = 0

        if plank.pos.y + plank.height > self.height:
            plank.pos.y = self.height - plank.height

    def update_ball(self, ball):
        ball_velocity = polar(ball.direction, ball.speed)
        ball.pos.add(ball_velocity)

        if ball.pos.x < 0:
            ball.pos.x = 0
        if ball.pos.x + ball.size / 2 > self.width:
            ball.pos.x = self.width - ball.size / 2
        if ball.pos.y < 0:
            ball.pos.y = 0
        if ball.pos.y + ball.size / 2 > self.height:
            ball.pos.y = self.height - ball.size / 2

    def tick(self, left_inpt=0, right_inpt=0):
        self.update_plank(self.left_plank, left_inpt)
        self.update_plank(self.right_plank, right_inpt)
        self.update_ball(self.ball)


if __name__ == '__main__':
    from renderer import Renderer
    from time import sleep

    fps = 30
    renderer = Renderer(1000, 1000, 'PONG')
    pong = PONG()

    while renderer.is_running:
        sleep(1 / fps)
        renderer.update()

        pong.left_plank.render(renderer)
        pong.right_plank.render(renderer)
        pong.ball.render(renderer)
        renderer.canvas.scale('all', 500, 500, 5, 5)

        print(pong.left_plank.pos.y)

        pong.tick(1, -1)
