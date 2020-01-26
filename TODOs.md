# TODO

**26.01.2020**
- 

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
