# TODO

**14.01.2020**

- [x] Pull out initial conditions from the code of the environment
   - Pull out the polygon defining the track (condition the model on the polygon)
   - [Code of the environment](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)
   - [Already forked and modified environment](https://github.com/hrc2da/CarRacing)
- [~] Scrape 1000-2000 sequences from CarRacing-v0
   - 1000 steps for each rollout
- [~] Preprocess in single `hdf5` file
- [ ] Train conditioned model
- [ ] _Make model probabilistic (might leave for next week)_
