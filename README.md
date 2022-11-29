# The Apogee to Apogee Path Sampler (AAPS)

**Reference**: C. Sherlock, S. Urbas and M. Ludkin: *The Apogee to Apogee Path Sampler*.

You will need the C++ library `Eigen` installed.

To compile: in the AAPS directory type

`./compAAPS`

Type

`./AAPS`

to see the options. For example, to run on the 10d test target for 10000 iterations, printing every 1000 iterations, using K=5, Wtype=3 and epsilon=1.2 use

`./AAPS 0 10000 1000 1 0 3 1.2 5`

Output appears in the `Output` directory.

To run for 10^4 iterations, printing output every 10^3, on a 40d logistic target with eccentricity of 20 using jittered scale parameters spaced approximately linearly between 1 and 20 with epsilon=1.2 and K=10 use

`./AAPS 10404020 10000 1000 1 0 3 1.2 10`

*Chris Sherlock*
*29/11/2022*
