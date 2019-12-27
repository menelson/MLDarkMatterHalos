The galaxy-halo connection with machine learning 

The goal of this project is to predict galaxy properties from the properties of the dark matter haloes in which they form. 

As dark matter haloes cannot be directly observed, this connection is difficult to determine observationally, but can be derived with empirical models. The catalogues below for 10 different redshifts contain different halo properties (e.g. halo mass, peak mass through time, growth rate) that should be used as features, and galaxy properties (e.g. stellar mass and star formation rate) that should be used as labels. They are in the HDF5 format that can be read in python with the h5py or the pandas packages. 

The following non-exhaustive list contains suggested steps and ideas to achieve reasonable results.

1) Load the table and divide between features and labels
2) Check which features correlate best with the labels
3) Scale the features (depending on the ML algorithms to between 0 and 1)
4) Split the data in training, validation, and test sets
5) Test different ML algorithms. You should use at least 3 commonly used methods, e.g. random forests, SGD, SVM, neural networks
6) Think about feature importance (using random forests)
7) Think about regularization
8) Think about dimensionality reduction (manifold learning)
9) Plot learning curves - when do you stop?
Determine the hyperparameters with the validation set
What is the final score on the test set for each algorithm?
