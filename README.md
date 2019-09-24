# Differentiable Simulation
[Learning to simulate Conway's Game of Life](./learning_conways_game_of_life.ipynb)

# Content
This repo contains materials for for my masters project on learning statefull simulations with deep differentiable models. The focus will be to train a neural network to be an end-to-end game.

## Setup
The abstraction for a video game I will look at is as follows:
 - black box that has a state
 - every tick the black box takes input
 - changes the state based on the input
 - and renders the frame of the game based on the state

This fits really well with the shape of the data that RNNs are designed to model so this woould naturally be my first experiment to try.

Concretely the game I am going to attempt to model is PONG. This game can be described with a small state simple rules. The task of the model would be to learn to manage the state of the game. To learn how to update it, based on the input, and how to use it to render every frame of the game.
I recon it will be interesting to analyze how different parts of the state influence the output.
