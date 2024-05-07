#include <stdio.h>
#include <stdint.h>
#include "../include/utils.cuh"
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

__device__ int done = 0;

__global__ void findNonce(char *block_content_final, int current_length, char *block_hash_final, char *difficulty, uint64_t *nonce) {
	char nonce_string[NONCE_SIZE];
	char block_hash[SHA256_HASH_SIZE];
	char block_content[BLOCK_SIZE];
	d_strcpy(block_content, block_content_final);

	int global_index = threadIdx.x + blockDim.x * blockIdx.x;
	int p = blockDim.x * gridDim.x;
	int n = MAX_NONCE;

	int start = global_index * (double)n / p;
	int end = (global_index + 1) * (double)n / p;

	for (int i = start; i < end; ++i) {
		if (done)
			break;

		intToString(i, nonce_string);
        d_strcpy(block_content + current_length, nonce_string);
        apply_sha256((BYTE *)block_content, d_strlen(block_content), (BYTE *)block_hash, 1);

        if (compare_hashes((BYTE *)block_hash, (BYTE *)difficulty) <= 0) {
			if (!atomicCAS(&done, 0, 1)) {
				d_strcpy(block_hash_final, block_hash);
				*nonce = i;
			}
            break;
        }

	}
}

int main(int argc, char **argv) {
	BYTE hashed_tx1[SHA256_HASH_SIZE], hashed_tx2[SHA256_HASH_SIZE], hashed_tx3[SHA256_HASH_SIZE], hashed_tx4[SHA256_HASH_SIZE],
			tx12[SHA256_HASH_SIZE * 2], tx34[SHA256_HASH_SIZE * 2], hashed_tx12[SHA256_HASH_SIZE], hashed_tx34[SHA256_HASH_SIZE],
			tx1234[SHA256_HASH_SIZE * 2], top_hash[SHA256_HASH_SIZE], block_content[BLOCK_SIZE];
	BYTE block_hash[SHA256_HASH_SIZE] = "0000000000000000000000000000000000000000000000000000000000000000";
	uint64_t nonce = 0;
	size_t current_length;

	// Top hash
	apply_sha256(tx1, strlen((const char*)tx1), hashed_tx1, 1);
	apply_sha256(tx2, strlen((const char*)tx2), hashed_tx2, 1);
	apply_sha256(tx3, strlen((const char*)tx3), hashed_tx3, 1);
	apply_sha256(tx4, strlen((const char*)tx4), hashed_tx4, 1);
	strcpy((char *)tx12, (const char *)hashed_tx1);
	strcat((char *)tx12, (const char *)hashed_tx2);
	apply_sha256(tx12, strlen((const char*)tx12), hashed_tx12, 1);
	strcpy((char *)tx34, (const char *)hashed_tx3);
	strcat((char *)tx34, (const char *)hashed_tx4);
	apply_sha256(tx34, strlen((const char*)tx34), hashed_tx34, 1);
	strcpy((char *)tx1234, (const char *)hashed_tx12);
	strcat((char *)tx1234, (const char *)hashed_tx34);
	apply_sha256(tx1234, strlen((const char*)tx34), top_hash, 1);

	// prev_block_hash + top_hash
	strcpy((char*)block_content, (const char*)prev_block_hash);
	strcat((char*)block_content, (const char*)top_hash);
	current_length = strlen((char*) block_content);

	char *dev_block_content, *dev_block_hash, *dev_difficulty;
	uint64_t *dev_nonce;

	cudaEvent_t start, stop;
	startTiming(&start, &stop);
	
	cudaMalloc((void **) &dev_block_content, BLOCK_SIZE);
	cudaMalloc((void **) &dev_block_hash, SHA256_HASH_SIZE);
	cudaMalloc((void **) &dev_difficulty, SHA256_HASH_SIZE);
	cudaMalloc((void **) &dev_nonce, sizeof(*dev_nonce));
	cudaMemcpy(dev_block_content, block_content, BLOCK_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_difficulty, DIFFICULTY, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);

	findNonce<<<50, 256>>>(dev_block_content, current_length, dev_block_hash, dev_difficulty, dev_nonce);

	cudaMemcpy(block_hash, dev_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)&nonce, dev_nonce, sizeof(*dev_nonce), cudaMemcpyDeviceToHost);

	float seconds = stopTiming(&start, &stop);
	printResult(block_hash, nonce, seconds);

	cudaFree(dev_block_content);
	cudaFree(dev_block_hash);
	cudaFree(dev_difficulty);
	cudaFree(dev_nonce);

	return 0;
}
