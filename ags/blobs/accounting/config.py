"""The blobs accounting experiment, defined once.

Every driver in this directory reads the same experiment, so it is defined here rather than in
whichever driver happened to be written first. Adding a dataset to the accounting study means
writing a sibling `config.py`, not copying a driver.

The dataset is deliberately hard for its size: seven of the eight coordinates carry inflated
nuisance variance, which raises the sample complexity while holding the Bayes ceiling fixed, so a
subsample-and-aggregate baseline cannot win for free.

`PRIVACY_STEPS` is the count that matters for accounting, and it is not the number of optimiser
updates. The minibatches form a public fixed partition, so a record is touched in exactly one
minibatch per epoch; adaptive parallel composition therefore charges one release per epoch, not one
per update. That is a property of the fixed-cyclic schedule, not sampling amplification.
"""

# data
N, D, ANISO, SEP = 200, 8, 5.0, 4.0

# learner: hinge loss, linear model, coordinate-clipped SGD
BATCH_SIZE, STEPS, GAMMA, LR, LAM = 20, 200, 0.05, 0.5, 0.0
PARAMETERS = D + 1

# accounting
PRIVACY_STEPS = STEPS // (N // BATCH_SIZE)  # one record-touching release per epoch
FREQUENCIES = (1, 2, 5, 10)  # MILP concretization frequencies with stored envelopes
EPSILONS = (1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100)
