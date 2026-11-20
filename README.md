# nedsurfcast

The following models are trained on: 
 - ridge regression (ridge)
 - ridge regression with history from previous timesteps + location features + fetch correction (ridge-hlf)
 - mlp (mlp)
 - mlp with history from previous timesteps + location features + fetch correction (mlp-hlf)
One file should measure performance by the skill between the model output and the raw GFS spectra that are available for K113 and EPL3
A table with the following skill percentages should be printed out at the end of the training run for each model: sqrt(E) | direction | Hm0 | Hm0 T<6 s | Hm0 6-10 s | Hm0 T>10 s.

Another file should use each station as a test station. Two folds per station group (even and odd months), each leaving out the group and its test months. Use a normalised error between the model prediction and the actual measurement, for example Hm0 NRMSE = RMSE of Hm0 / RMS of the measured Hm0. structure the output in the same way as the first training script: squaredroot E, Direction, Hm0, Hm < 6 s, 6 s =< Hm <= 10 s, Hm > 10 s.1, averaged over all folds for every model per row. Also, save a .csv with the same columns for every model for every test fold that holdouts 1 station group, because it gives information about potential problems at individual stations.