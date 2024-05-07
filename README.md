Copyright 2024 Maria Sfiraiala (maria.sfiraiala@stud.acs.upb.ro)

# CUDA Bitcoin Proof of Work Consensus Algorithm - Project2

## Description

The project aims to parallelize on the GPU the creation of nonces (random numbers computed and checked by Bitcoin miners) that should comply with certain rules.

In order to get the best computation time, I've decided to use the "start-end" method learned in a [previous uni lab, for the Parallel and Distributed Algorithms course](https://mobylab.docs.crescdi.pub.ro/docs/parallelAndDistributed/laboratory1/exercises).
The idea behind this method centers around dividing the loop between all the threads and providing every worker with a portion that it has exclusive access to.

To decide which portion belongs to which thread, I had to:

1. compute the thread index using the formula from the CUDA labs:

    ```C
	int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    ```

1. get the total number of threads:

    ```C
	int p = blockDim.x * gridDim.x;
    ```

1. get the number of iterations from the sequential loop:

    ```C
	int n = MAX_NONCE;
    ```

1. find start and stop using the formula from the previous semester course:

    ```C
	int start = global_index * (double)n / p;
	int end = (global_index + 1) * (double)n / p;
    ```

The loop is now ready to be divided and the nonce and block hash ready to be computed using the same approach as in the sequential variant (test5).

What remains to be done is to allocate memory for the parameters needed by the function, copy the contents from the source variables (`block_content` and `difficulty`) and copy back the information from the destination variables (`block_hash` and `nonce`).

I also determined (empirically) that having 50 blocks and 256 threads per block is good enough for the execution time, so I rolled with that. 

## Results

I've run the program 20 times in order to get the average time.
The result was 0.08 seconds, with the intermidate results being:

```text
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.09
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.20
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.15
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.21
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.21
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.07
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.05
00000466c22e6ee57f6ec5a8122e67f82a381499a4b3069869819639bb22a2ee,515800,0.11
```

Previous implementations of mine had some differences that made them faster / slower:

1. first implementation: I didn't break the loop that searches for the nonces inside the other threads.
As a result, the run lasted for ~2 seconds, which was kinda bad, therefore I decided to add the global flag `done` that would signal **all** the threads that they should stop.

1. second implementation: the global flag `done` was not atomically changed, with the possibility of 2 threads attempting to set it and write the result in the memory zone reserved for it (that is accessible by all threads).
By protecting the "critical zone" with an atomic compare and swap, I increased the execution time from the 0.05 average to the present 0.08.

## Observations Regarding The Project

A fine, easy enough project, I just wish the team didn't consider the holyday week for the project duration.
I was sick, I had to do chores inside the house and I had to celebrate Easter, all while working for this assignment.
Sad.

At least, good job at coming up with something truly helpful as the idea behind the project <3!
