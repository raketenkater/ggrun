#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static int current_device;

int cudaSetDevice(int device) {
    current_device = device;
    return 0;
}

int cudaGetDevice(int * device) {
    *device = current_device;
    return 0;
}

int cudaMalloc(void ** ptr, size_t size) {
    *ptr = malloc(size);
    return *ptr == NULL ? 2 : 0;
}

int cudaFree(void * ptr) {
    free(ptr);
    return 0;
}

int cudaMallocAsync(void ** ptr, size_t size, void * stream) {
    (void) stream;
    return cudaMalloc(ptr, size);
}

int cudaFreeAsync(void * ptr, void * stream) {
    (void) stream;
    return cudaFree(ptr);
}

int cudaHostAlloc(void ** ptr, size_t size, unsigned int flags) {
    (void) flags;
    *ptr = malloc(size);
    return *ptr == NULL ? 2 : 0;
}

int cudaFreeHost(void * ptr) {
    free(ptr);
    return 0;
}
