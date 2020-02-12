# TODO

**04.02.2020**
- [~] Read [DNC paper](https://www.gwern.net/docs/rl/2016-graves.pdf)
- [ ] Read [IMPROVING DIFFERENTIABLE NEURAL COMPUTERS - Schmidhuber](https://openreview.net/pdf?id=HyGEM3C9KQ)
- [~] Try implementation
   - [DeepMind/DNC](https://github.com/deepmind/dnc)
   - [x] [pytorch - xdever/dnc](https://github.com/xdever/dnc)
      - Kinda slow
- [ ] Try to implement it yourself? (or at least understand other people's implementation)
   - [CUDA RNN in pytorch](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/)
      - Useful example of `jit.ScriptModule`
      - Fusing CUDA kernels
      - **For loop unrolling** - sounds like a big deal for rnn
   - [Introduction to torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
      - Useful examples of scripting, tracing, saving and loading

**26.01.2020**
- smth smth
- waste time with cppn training
- decide to work with DNC

**14.01.2020**
- [x] Pull out initial conditions from the code of the environment
   - Pull out the polygon defining the track (condition the model on the polygon)
   - [Code of the environment](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)
   - [Already forked and modified environment](https://github.com/hrc2da/CarRacing)
- [x] Scrape 1500 sequences from CarRacing-v0
   - 1000 steps for each rollout
- [x] Preprocess in single `hdf5` file
   - [Deep Learning . Training with huge amount of data .PART1](https://medium.com/@cristianzantedeschi/deep-learning-regression-feeding-huge-amount-of-data-to-gpu-performance-considerations-2934d32ab315)
   - [HDF5 Advantages: Organization, flexibility, interoperability](https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr)
- [x] Train conditioned model
- [x] Make model probabilistic (might leave for next 
