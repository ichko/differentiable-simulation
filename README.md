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

Concretely the game I am going to attempt to model is PONG. This game can be described with a small state and simple rules. The task of the model would be to learn to manage the state of the game. To learn how to update it, based on the input, and how to use it to render every frame of the game.
I recon it will be interesting to analyze how different parts of the state influence the output.

## Ideas for research
 - Pyramidaly stacked RNN
     - Let high level RNN cells observe only parts of the sequence (e.g. only every second output of the RNN below it).
     - Idea being that these cells will hold long term state that is updated less frequently. 
     - It will be easier for the gradient from the front to propagate to the back since the higher level sequences will be shorter.
 - Runtime feature detectors
     - Train two networks, the first one learns to predict the detectors of the second. Both of the networks are ran on the same input. The idea is that We would have input specific featuree detectors.
     - Polynomial type of interaction of the input with the network?
     - This idea could be benificial in the case of the Learned GameEngine - generating (simulation) network based on the embedding of the simulation (the game).
     - This might be beneficial in the case of the vanishing gradient problem.
 - Progressive sequence growth
     - Start with short sequences and ingrease the length in the process of learning.
     - Hypothesis is that this might speed up training for longer sequences.
     - Simmilar to the idea of ProGAN.
 - Internally stacked RNN.
     - State<t> is complex DNN of input<t> and state<t-1>
     - Might play well with the pyramidally stacked RNN.

## TODOs

 - Optimize input pipeline