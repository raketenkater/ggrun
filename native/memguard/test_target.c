#include <stddef.h>
#include <stdio.h>

int cudaSetDevice(int device);
int cudaMalloc(void ** ptr, size_t size);
int cudaFree(void * ptr);
int cudaMallocAsync(void ** ptr, size_t size, void * stream);
int cudaFreeAsync(void * ptr, void * stream);
int cudaHostAlloc(void ** ptr, size_t size, unsigned int flags);

int main(void) {
    void * first = NULL;
    void * denied = NULL;
    void * pinned = NULL;
    void * async_alloc = NULL;
    if (cudaSetDevice(0) != 0) return 10;
    if (cudaMalloc(&first, 512 * 1024) != 0 || first == NULL) return 11;
    if (cudaMalloc(&denied, 768 * 1024) != 2 || denied != NULL) return 12;
    if (cudaHostAlloc(&pinned, 4096, 0) != 2 || pinned != NULL) return 13;
    if (cudaFree(first) != 0) return 14;
    if (cudaMallocAsync(&async_alloc, 768 * 1024, NULL) != 0 || async_alloc == NULL) return 15;
    if (cudaFreeAsync(async_alloc, NULL) != 0) return 16;
    puts("memguard synthetic target passed");
    return 0;
}
