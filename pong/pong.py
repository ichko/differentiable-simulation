import math
from renderer import Renderer
from time import sleep, time
from random import uniform, random as rand


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vec):
        self.x += vec.x
        self.y += vec.y

        return self

    @property
    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    @property
    def noramalized(self):
        len = self.length
        return Vector(self.x / len, self.y / len)

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
        self.speed = 1

    def render(self, renderer):
        renderer.rect(
            self.pos.x, self.pos.y,
            self.width, self.height
        )


class Ball:
    def __init__(self, x, y, size):
        self.pos = vec(x, y)
        self.velocity = polar(uniform(0, math.pi), 2)
        self.size = size

    def render(self, renderer):
        renderer.rect(
            self.pos.x - self.size,
            self.pos.y + self.size,
            self.size * 2,
            self.size * 2
        )


class PONG:
    def __init__(self, w, h, pw, ph, bs):
        self.width = w
        self.height = h

        self.plank_width = pw
        self.plank_height = ph
        self.ball_size = bs
        self.game_over = False

        self.left_plank = Plank(
            -self.width / 2, 0,
            self.plank_width, self.plank_height
        )

        self.right_plank = Plank(
            self.width / 2 - self.plank_width, 0,
            self.plank_width, self.plank_height
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

        left_wall = -self.width / 2 + (ball.size + 1) + self.plank_width
        right_wall = self.width / 2 - (ball.size + 1) - self.plank_width

        top_wall = self.height / 2 - ball.size
        bottom_wall = -self.height / 2 + ball.size + 1

        left_h_constraint = ball.pos.y - ball.size < self.left_plank.pos.y and \
            ball.pos.y + ball.size > self.left_plank.pos.y - self.left_plank.height

        right_h_constraint = ball.pos.y - ball.size < self.right_plank.pos.y and \
            ball.pos.y + ball.size > self.right_plank.pos.y - self.right_plank.height

        if not left_h_constraint and ball.pos.x < left_wall or \
           not right_h_constraint and ball.pos.x > right_wall:
            self.game_over = True

        if ball.pos.x > right_wall:
            ball.pos.x = right_wall
            ball.velocity.x *= -1

        if ball.pos.x < left_wall:
            ball.pos.x = left_wall
            ball.velocity.x *= -1

        if ball.pos.y > top_wall:
            ball.pos.y = top_wall
            ball.velocity.y *= -1

        if ball.pos.y < bottom_wall:
            ball.pos.y = bottom_wall
            ball.velocity.y *= -1

    def tick(self, left_inpt=0, right_inpt=0):
        self.update_plank(self.left_plank, left_inpt)
        self.update_plank(self.right_plank, right_inpt)
        self.update_ball(self.ball)


if __name__ == '__main__':

    fps = 1000
    pong = PONG(
        50, 50,
        5, 15,
        2
    )

    @Renderer(50, 50)
    def loop(R):
        sleep(1 / fps)

        pong.left_plank.render(R)
        pong.right_plank.render(R)
        pong.ball.render(R)

        left_dir_picker = pong.ball.pos.y - pong.left_plank.pos.y + pong.plank_height / 2 \
            if pong.ball.pos.x <= 0 else math.sin(R.f / 10)

        right_dir_picker = pong.ball.pos.y - pong.right_plank.pos.y + pong.plank_height / 2 \
            if pong.ball.pos.x >= 0 else math.sin(R.f / 10)

        left_plank_dir = math.copysign(1, left_dir_picker)
        right_plank_dir = math.copysign(1, right_dir_picker)

        # if not pong.game_over:
        pong.tick(left_plank_dir, right_plank_dir)
