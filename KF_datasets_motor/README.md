Kalman Filter Datasets
======================

Contains three brain datasets:
1) Motor cortex from non-human primate
2) Somatosensory cortex from non-human primate (soma)
3) Hippocampus from a rat (hc)

Motor dimensions: 6 states, 164 neurons
Soma dimensions: 6 states, 52 neurons
HC dimensions: 6 states, 46 neurons

Motor number of time samples: 3793
Soma number of time samples: 9199
   HC number of time samples: 4176

Notes:
======
1) Each matrix of data is saved in a different file.
2) Different file for each dataset.
3) All model matrices are saved in flattened arrays row major. It means row 0 then row 1, row 2 and so on...
4) The samples from the neurons and real states data are saved in flattened arrays one vector after the other. Mesurements vec 0, measurement vec 1 and so on...
5) The naming of the matrices are a bit different, but this is how they match coompared to the   definition:
   initial vector x --> initial_state_array
   initial matrix P --> initialize as a zero matrix
   Matrix F --> A array
   Matrix Q --> W array
   Matrix R --> Q array
   Matrix H --> H array
   measurements --> measurements array
   actual motor kinematics --> real array
   reference Kalman filter predictions --> prediction aray

