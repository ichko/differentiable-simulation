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
    def __init__(self, x, y, width, height):
        self.pos = vec(x, y)
        self.width = width
        self.height = height
        self.speed = 2

    def render(self, renderer):
        renderer.rect(
            self.pos.x, self.pos.y,
            self.width, self.height,
            fill='#fff'
        )


class Ball:
    def __init__(self, x, y, size):
        self.pos = vec(x, y)
        self.velocity = vec(0.5, 1)
        self.size = size

    def render(self, renderer):
        renderer.arc(self.pos.x, self.pos.y, self.size, fill='#fff')


class PONG:
    def __init__(self):
        self.width = 20
        self.height = 20

        self.plank_width = 2
        self.plank_height = 5
        self.ball_size = 2

        self.left_plank = Plank(
            -self.width / 2, 0, self.plank_width, self.plank_height
        )

        self.right_plank = Plank(
            self.width / 2 - self.plank_width, 0, self.plank_width, self.plank_height
        )

        self.ball = Ball(0, 0, self.ball_size)

    def update_plank(self, plank, inpt):
        plank.pos.y += inpt * plank.speed

        if plank.pos.y > self.height / 2:
            plank.pos.y = self.height / 2

        if plank.pos.y < -self.height / 2 + plank.height:
            plank.pos.y = -self.height / 2 + plank.height

    def update_ball(self, ball):
        ball.pos.add(ball.velocity)

        if ball.pos.x > self.width / 2 - ball.size - self.plank_width:
            ball.pos.x = self.width / 2 - ball.size - self.plank_width
            ball.velocity.x *= -1

        if ball.pos.x < -self.width / 2 + ball.size + self.plank_width:
            ball.pos.x = -self.width / 2 + ball.size + self.plank_width
            ball.velocity.x *= -1

        if ball.pos.y > self.height / 2 - ball.size:
            ball.pos.y = self.height / 2 - ball.size
            ball.velocity.y *= -1

        if ball.pos.y < -self.height / 2 + ball.size:
            ball.pos.y = -self.height / 2 + ball.size
            ball.velocity.y *= -1

    def tick(self, left_inpt=0, right_inpt=0):
        self.update_plank(self.left_plank, left_inpt)
        self.update_plank(self.right_plank, right_inpt)
        self.update_ball(self.ball)


if __name__ == '__main__':
    from renderer import Renderer
    from time import sleep
    from random import random

    fps = 60
    t = 0
    W, H = 1000, 1000
    renderer = Renderer(W, H, 'PONG')
    pong = PONG()

    while renderer.is_running:
        t += 0.1
        sleep(1 / fps)
        renderer.update()

        renderer.rect(-pong.width / 2, pong.width / 2,
                      pong.width, pong.height, fill='#444')
        pong.left_plank.render(renderer)
        pong.right_plank.render(renderer)
        pong.ball.render(renderer)
        renderer.canvas.scale(
            'all', W / 2, H / 2,
            W / pong.width - 0.5, H / pong.height - 0.5
        )

        pong.tick(math.sin(t), math.cos(t))
