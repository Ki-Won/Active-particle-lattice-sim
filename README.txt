This code simulates active particles on square lattice.
The code is parallelized using GPU.

1D rec_EP: 
The thread is allocated for each particle.
In 1D periodic space, simulate non-interacting Run-and-Tumble particles(RTPs). 
The applied potential is asymmetric inducing rectified current of RTPs.
The code measures the rectified current and entropy production.

2D pressure:
The thread is allocated for each particle.
1n 2D space, simulate non-interacting Active-Brownina particles(ABPs).
The space is periodic in y-direction and stiff potential is applied at both ends in x-direction.
The code measures exerted pressure to the potential from ABPs.
