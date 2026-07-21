#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// CUDA runtime and driver APIs use integer result codes. Both define OOM as 2.
#define GGRUN_CUDA_OOM 2
#define GGRUN_CUDA_UNKNOWN 999
#define GGRUN_MAX_GPUS 32
#define GGRUN_MAX_ALLOCS 65536

enum allocation_kind {
    ALLOC_DEVICE,
    ALLOC_MANAGED,
    ALLOC_PINNED,
    ALLOC_MLOCK,
};

struct allocation_record {
    uintptr_t ptr;
    size_t size;
    int device;
    enum allocation_kind kind;
    int used;
};

static pthread_mutex_t state_mu = PTHREAD_MUTEX_INITIALIZER;
static struct allocation_record records[GGRUN_MAX_ALLOCS];
static uint64_t device_active[GGRUN_MAX_GPUS];
static uint64_t device_peak[GGRUN_MAX_GPUS];
static uint64_t device_limit[GGRUN_MAX_GPUS];
static uint64_t host_active;
static uint64_t host_peak;
static uint64_t host_limit;
static int device_limit_set[GGRUN_MAX_GPUS];
static int host_limit_set;
static int log_fd = -1;
static __thread int runtime_device;
static __thread int forwarding_depth;

static const char * kind_name(enum allocation_kind kind) {
    switch (kind) {
        case ALLOC_DEVICE: return "device";
        case ALLOC_MANAGED: return "managed";
        case ALLOC_PINNED: return "pinned";
        case ALLOC_MLOCK: return "mlock";
    }
    return "unknown";
}

static uint64_t parse_mb(const char * value) {
    if (value == NULL || *value == '\0') return 0;
    char * end = NULL;
    unsigned long long mb = strtoull(value, &end, 10);
    if (end == value || *end != '\0') return 0;
    return (uint64_t) mb * 1024ULL * 1024ULL;
}

static void emit_event(
        const char * phase,
        const char * api,
        enum allocation_kind kind,
        int device,
        size_t bytes,
        uint64_t active,
        uint64_t peak,
        uint64_t limit,
        int result,
        uintptr_t ptr) {
    if (log_fd < 0) return;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    char line[768];
    int n = snprintf(line, sizeof(line),
        "{\"type\":\"allocation\",\"phase\":\"%s\",\"api\":\"%s\","
        "\"kind\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"time_ns\":%lld,"
        "\"device\":%d,\"bytes\":%zu,\"active_bytes\":%llu,"
        "\"peak_bytes\":%llu,\"limit_bytes\":%llu,\"result\":%d,"
        "\"pointer\":%llu}\n",
        phase, api, kind_name(kind), (long) getpid(), (long) gettid(),
        (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec,
        device, bytes, (unsigned long long) active, (unsigned long long) peak,
        (unsigned long long) limit, result, (unsigned long long) ptr);
    if (n > 0) {
        size_t count = (size_t) n < sizeof(line) ? (size_t) n : sizeof(line) - 1;
        ssize_t written = write(log_fd, line, count);
        (void) written;
    }
}

static void emit_loaded(void) {
    if (log_fd < 0) return;
    char line[256];
    int n = snprintf(line, sizeof(line),
        "{\"type\":\"guard\",\"event\":\"loaded\",\"pid\":%ld,"
        "\"schema_version\":1}\n", (long) getpid());
    if (n > 0) {
        ssize_t written = write(log_fd, line, (size_t) n);
        (void) written;
    }
}

static void parse_gpu_limits(const char * value) {
    if (value == NULL || *value == '\0') return;
    char copy[1024];
    snprintf(copy, sizeof(copy), "%s", value);
    char * save = NULL;
    char * token = strtok_r(copy, ",", &save);
    int index = 0;
    while (token != NULL && index < GGRUN_MAX_GPUS) {
        device_limit[index] = parse_mb(token);
        device_limit_set[index] = 1;
        token = strtok_r(NULL, ",", &save);
        index++;
    }
}

__attribute__((constructor)) static void memguard_init(void) {
    const char * path = getenv("GGRUN_MEMGUARD_LOG");
    if (path != NULL && *path != '\0') {
        log_fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    }
    parse_gpu_limits(getenv("GGRUN_MEMGUARD_GPU_LIMITS_MB"));
    const char * host = getenv("GGRUN_MEMGUARD_PINNED_LIMIT_MB");
    if (host != NULL && *host != '\0') {
        host_limit = parse_mb(host);
        host_limit_set = 1;
    }
    emit_loaded();
}

__attribute__((destructor)) static void memguard_fini(void) {
    if (log_fd >= 0) close(log_fd);
}

static void * next_symbol(const char * name) {
    dlerror();
    return dlsym(RTLD_NEXT, name);
}

static int current_driver_device(void) {
    typedef int (*fn_t)(int *);
    fn_t fn = (fn_t) next_symbol("cuCtxGetDevice");
    int device = runtime_device;
    if (fn != NULL) {
        forwarding_depth++;
        int found = device;
        if (fn(&found) == 0 && found >= 0 && found < GGRUN_MAX_GPUS) device = found;
        forwarding_depth--;
    }
    return device;
}

static int allocation_allowed(enum allocation_kind kind, int device, size_t size,
        const char * api) {
    pthread_mutex_lock(&state_mu);
    uint64_t active = 0;
    uint64_t peak = 0;
    uint64_t limit = 0;
    int limited = 0;
    if (kind == ALLOC_DEVICE || kind == ALLOC_MANAGED) {
        if (device < 0 || device >= GGRUN_MAX_GPUS) device = 0;
        active = device_active[device];
        peak = device_peak[device];
        limit = device_limit[device];
        limited = device_limit_set[device];
    } else {
        active = host_active;
        peak = host_peak;
        limit = host_limit;
        limited = host_limit_set;
    }
    int allowed = !limited || size <= limit - (active > limit ? limit : active);
    if (!allowed) emit_event("denied", api, kind, device, size, active, peak, limit,
            GGRUN_CUDA_OOM, 0);
    pthread_mutex_unlock(&state_mu);
    return allowed;
}

static void record_allocation(uintptr_t ptr, enum allocation_kind kind, int device,
        size_t size, const char * api, int result) {
    pthread_mutex_lock(&state_mu);
    uint64_t active = 0;
    uint64_t peak = 0;
    uint64_t limit = 0;
    if (result == 0 && ptr != 0) {
        for (size_t i = 0; i < GGRUN_MAX_ALLOCS; i++) {
            if (!records[i].used) {
                records[i] = (struct allocation_record) {
                    .ptr = ptr, .size = size, .device = device, .kind = kind, .used = 1,
                };
                break;
            }
        }
        if (kind == ALLOC_DEVICE || kind == ALLOC_MANAGED) {
            if (device < 0 || device >= GGRUN_MAX_GPUS) device = 0;
            device_active[device] += size;
            if (device_active[device] > device_peak[device]) device_peak[device] = device_active[device];
        } else {
            host_active += size;
            if (host_active > host_peak) host_peak = host_active;
        }
    }
    if (kind == ALLOC_DEVICE || kind == ALLOC_MANAGED) {
        if (device < 0 || device >= GGRUN_MAX_GPUS) device = 0;
        active = device_active[device];
        peak = device_peak[device];
        limit = device_limit[device];
    } else {
        active = host_active;
        peak = host_peak;
        limit = host_limit;
    }
    emit_event("result", api, kind, device, size, active, peak, limit, result, ptr);
    pthread_mutex_unlock(&state_mu);
}

static void record_free(uintptr_t ptr, const char * api, int result) {
    pthread_mutex_lock(&state_mu);
    for (size_t i = 0; i < GGRUN_MAX_ALLOCS; i++) {
        if (!records[i].used || records[i].ptr != ptr) continue;
        struct allocation_record rec = records[i];
        if (result == 0) {
            records[i].used = 0;
            if (rec.kind == ALLOC_DEVICE || rec.kind == ALLOC_MANAGED) {
                int d = rec.device >= 0 && rec.device < GGRUN_MAX_GPUS ? rec.device : 0;
                device_active[d] = device_active[d] >= rec.size ? device_active[d] - rec.size : 0;
                emit_event("free", api, rec.kind, d, rec.size, device_active[d],
                        device_peak[d], device_limit[d], result, ptr);
            } else {
                host_active = host_active >= rec.size ? host_active - rec.size : 0;
                emit_event("free", api, rec.kind, rec.device, rec.size, host_active,
                        host_peak, host_limit, result, ptr);
            }
        }
        break;
    }
    pthread_mutex_unlock(&state_mu);
}

int cudaSetDevice(int device) {
    typedef int (*fn_t)(int);
    fn_t fn = (fn_t) next_symbol("cudaSetDevice");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    int result = fn(device);
    if (result == 0) runtime_device = device;
    return result;
}

static int runtime_allocate(const char * name, void ** ptr, size_t size,
        enum allocation_kind kind, unsigned int flags) {
    typedef int (*malloc_fn)(void **, size_t);
    typedef int (*flag_fn)(void **, size_t, unsigned int);
    void * symbol = next_symbol(name);
    if (symbol == NULL) return GGRUN_CUDA_UNKNOWN;
    int device = kind == ALLOC_DEVICE || kind == ALLOC_MANAGED ? runtime_device : -1;
    if (forwarding_depth == 0 && !allocation_allowed(kind, device, size, name)) {
        if (ptr != NULL) *ptr = NULL;
        return GGRUN_CUDA_OOM;
    }
    forwarding_depth++;
    int result = flags == UINT32_MAX
        ? ((malloc_fn) symbol)(ptr, size)
        : ((flag_fn) symbol)(ptr, size, flags);
    forwarding_depth--;
    if (forwarding_depth == 0) record_allocation(ptr != NULL ? (uintptr_t) *ptr : 0,
            kind, device, size, name, result);
    return result;
}

int cudaMalloc(void ** ptr, size_t size) {
    return runtime_allocate("cudaMalloc", ptr, size, ALLOC_DEVICE, UINT32_MAX);
}

int cudaMallocManaged(void ** ptr, size_t size, unsigned int flags) {
    return runtime_allocate("cudaMallocManaged", ptr, size, ALLOC_MANAGED, flags);
}

static int runtime_allocate_async(const char * name, void ** ptr, size_t size,
        void * pool, void * stream) {
    typedef int (*async_fn)(void **, size_t, void *);
    typedef int (*pool_fn)(void **, size_t, void *, void *);
    void * symbol = next_symbol(name);
    if (symbol == NULL) return GGRUN_CUDA_UNKNOWN;
    int device = runtime_device;
    if (forwarding_depth == 0 && !allocation_allowed(ALLOC_DEVICE, device, size, name)) {
        if (ptr != NULL) *ptr = NULL;
        return GGRUN_CUDA_OOM;
    }
    forwarding_depth++;
    int result = pool == NULL
        ? ((async_fn) symbol)(ptr, size, stream)
        : ((pool_fn) symbol)(ptr, size, pool, stream);
    forwarding_depth--;
    if (forwarding_depth == 0) record_allocation(ptr != NULL ? (uintptr_t) *ptr : 0,
            ALLOC_DEVICE, device, size, name, result);
    return result;
}

int cudaMallocAsync(void ** ptr, size_t size, void * stream) {
    return runtime_allocate_async("cudaMallocAsync", ptr, size, NULL, stream);
}

int cudaMallocFromPoolAsync(void ** ptr, size_t size, void * pool, void * stream) {
    return runtime_allocate_async("cudaMallocFromPoolAsync", ptr, size, pool, stream);
}

int cudaHostAlloc(void ** ptr, size_t size, unsigned int flags) {
    return runtime_allocate("cudaHostAlloc", ptr, size, ALLOC_PINNED, flags);
}

int cudaMallocHost(void ** ptr, size_t size) {
    return runtime_allocate("cudaMallocHost", ptr, size, ALLOC_PINNED, UINT32_MAX);
}

int cudaHostRegister(void * ptr, size_t size, unsigned int flags) {
    typedef int (*fn_t)(void *, size_t, unsigned int);
    fn_t fn = (fn_t) next_symbol("cudaHostRegister");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth == 0 && !allocation_allowed(ALLOC_PINNED, -1, size, "cudaHostRegister")) return GGRUN_CUDA_OOM;
    forwarding_depth++;
    int result = fn(ptr, size, flags);
    forwarding_depth--;
    if (forwarding_depth == 0) record_allocation((uintptr_t) ptr, ALLOC_PINNED, -1,
            size, "cudaHostRegister", result);
    return result;
}

int cudaFree(void * ptr) {
    typedef int (*fn_t)(void *);
    fn_t fn = (fn_t) next_symbol("cudaFree");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    forwarding_depth++;
    int result = fn(ptr);
    forwarding_depth--;
    if (forwarding_depth == 0) record_free((uintptr_t) ptr, "cudaFree", result);
    return result;
}

int cudaFreeAsync(void * ptr, void * stream) {
    typedef int (*fn_t)(void *, void *);
    fn_t fn = (fn_t) next_symbol("cudaFreeAsync");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    forwarding_depth++;
    int result = fn(ptr, stream);
    forwarding_depth--;
    // The allocation remains live until the stream reaches this operation.
    // Keeping it charged is conservative; releasing it here could permit a
    // later allocation while both are still resident on the device.
    return result;
}

int cudaFreeHost(void * ptr) {
    typedef int (*fn_t)(void *);
    fn_t fn = (fn_t) next_symbol("cudaFreeHost");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    forwarding_depth++;
    int result = fn(ptr);
    forwarding_depth--;
    if (forwarding_depth == 0) record_free((uintptr_t) ptr, "cudaFreeHost", result);
    return result;
}

int cudaHostUnregister(void * ptr) {
    typedef int (*fn_t)(void *);
    fn_t fn = (fn_t) next_symbol("cudaHostUnregister");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    forwarding_depth++;
    int result = fn(ptr);
    forwarding_depth--;
    if (forwarding_depth == 0) record_free((uintptr_t) ptr, "cudaHostUnregister", result);
    return result;
}

static int driver_allocate(const char * name, uint64_t * ptr, size_t size,
        enum allocation_kind kind, unsigned int flags) {
    typedef int (*alloc_fn)(uint64_t *, size_t);
    typedef int (*managed_fn)(uint64_t *, size_t, unsigned int);
    void * symbol = next_symbol(name);
    if (symbol == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) {
        return flags == UINT32_MAX
            ? ((alloc_fn) symbol)(ptr, size)
            : ((managed_fn) symbol)(ptr, size, flags);
    }
    int device = current_driver_device();
    if (!allocation_allowed(kind, device, size, name)) {
        if (ptr != NULL) *ptr = 0;
        return GGRUN_CUDA_OOM;
    }
    forwarding_depth++;
    int result = flags == UINT32_MAX
        ? ((alloc_fn) symbol)(ptr, size)
        : ((managed_fn) symbol)(ptr, size, flags);
    forwarding_depth--;
    record_allocation(ptr != NULL ? (uintptr_t) *ptr : 0, kind, device, size, name, result);
    return result;
}

int cuMemAlloc_v2(uint64_t * ptr, size_t size) {
    return driver_allocate("cuMemAlloc_v2", ptr, size, ALLOC_DEVICE, UINT32_MAX);
}

int cuMemAllocManaged(uint64_t * ptr, size_t size, unsigned int flags) {
    return driver_allocate("cuMemAllocManaged", ptr, size, ALLOC_MANAGED, flags);
}

static int driver_allocate_async(const char * name, uint64_t * ptr, size_t size,
        uint64_t pool, void * stream) {
    typedef int (*async_fn)(uint64_t *, size_t, void *);
    typedef int (*pool_fn)(uint64_t *, size_t, uint64_t, void *);
    void * symbol = next_symbol(name);
    if (symbol == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) {
        return pool == 0
            ? ((async_fn) symbol)(ptr, size, stream)
            : ((pool_fn) symbol)(ptr, size, pool, stream);
    }
    int device = current_driver_device();
    if (!allocation_allowed(ALLOC_DEVICE, device, size, name)) {
        if (ptr != NULL) *ptr = 0;
        return GGRUN_CUDA_OOM;
    }
    forwarding_depth++;
    int result = pool == 0
        ? ((async_fn) symbol)(ptr, size, stream)
        : ((pool_fn) symbol)(ptr, size, pool, stream);
    forwarding_depth--;
    record_allocation(ptr != NULL ? (uintptr_t) *ptr : 0, ALLOC_DEVICE, device,
            size, name, result);
    return result;
}

int cuMemAllocAsync(uint64_t * ptr, size_t size, void * stream) {
    return driver_allocate_async("cuMemAllocAsync", ptr, size, 0, stream);
}

int cuMemAllocFromPoolAsync(uint64_t * ptr, size_t size, uint64_t pool, void * stream) {
    return driver_allocate_async("cuMemAllocFromPoolAsync", ptr, size, pool, stream);
}

int cuMemCreate(uint64_t * handle, size_t size, const void * prop, uint64_t flags) {
    typedef int (*fn_t)(uint64_t *, size_t, const void *, uint64_t);
    fn_t fn = (fn_t) next_symbol("cuMemCreate");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) return fn(handle, size, prop, flags);
    int device = current_driver_device();
    if (!allocation_allowed(ALLOC_DEVICE, device, size, "cuMemCreate")) {
        if (handle != NULL) *handle = 0;
        return GGRUN_CUDA_OOM;
    }
    forwarding_depth++;
    int result = fn(handle, size, prop, flags);
    forwarding_depth--;
    record_allocation(handle != NULL ? (uintptr_t) *handle : 0, ALLOC_DEVICE,
            device, size, "cuMemCreate", result);
    return result;
}

int cuMemFree_v2(uint64_t ptr) {
    typedef int (*fn_t)(uint64_t);
    fn_t fn = (fn_t) next_symbol("cuMemFree_v2");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) return fn(ptr);
    forwarding_depth++;
    int result = fn(ptr);
    forwarding_depth--;
    record_free((uintptr_t) ptr, "cuMemFree_v2", result);
    return result;
}

int cuMemFreeAsync(uint64_t ptr, void * stream) {
    typedef int (*fn_t)(uint64_t, void *);
    fn_t fn = (fn_t) next_symbol("cuMemFreeAsync");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) return fn(ptr, stream);
    forwarding_depth++;
    int result = fn(ptr, stream);
    forwarding_depth--;
    return result;
}

int cuMemRelease(uint64_t handle) {
    typedef int (*fn_t)(uint64_t);
    fn_t fn = (fn_t) next_symbol("cuMemRelease");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) return fn(handle);
    forwarding_depth++;
    int result = fn(handle);
    forwarding_depth--;
    record_free((uintptr_t) handle, "cuMemRelease", result);
    return result;
}

int cuMemHostAlloc(void ** ptr, size_t size, unsigned int flags) {
    return runtime_allocate("cuMemHostAlloc", ptr, size, ALLOC_PINNED, flags);
}

int cuMemFreeHost(void * ptr) {
    typedef int (*fn_t)(void *);
    fn_t fn = (fn_t) next_symbol("cuMemFreeHost");
    if (fn == NULL) return GGRUN_CUDA_UNKNOWN;
    if (forwarding_depth > 0) return fn(ptr);
    forwarding_depth++;
    int result = fn(ptr);
    forwarding_depth--;
    record_free((uintptr_t) ptr, "cuMemFreeHost", result);
    return result;
}

int mlock(const void * addr, size_t len) {
    typedef int (*fn_t)(const void *, size_t);
    fn_t fn = (fn_t) next_symbol("mlock");
    if (fn == NULL) { errno = ENOSYS; return -1; }
    if (!allocation_allowed(ALLOC_MLOCK, -1, len, "mlock")) { errno = ENOMEM; return -1; }
    int result = fn(addr, len);
    record_allocation((uintptr_t) addr, ALLOC_MLOCK, -1, len, "mlock", result == 0 ? 0 : errno);
    return result;
}

int munlock(const void * addr, size_t len) {
    typedef int (*fn_t)(const void *, size_t);
    fn_t fn = (fn_t) next_symbol("munlock");
    if (fn == NULL) { errno = ENOSYS; return -1; }
    int result = fn(addr, len);
    if (result == 0) record_free((uintptr_t) addr, "munlock", 0);
    return result;
}
