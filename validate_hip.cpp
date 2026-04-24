#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void hello_gfx906(float* out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    out[i] = (float)i * 2.0f;
}

int main() {
    // Check device count
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    if (err != hipSuccess || deviceCount == 0) {
        fprintf(stderr, "FAIL: hipGetDeviceCount: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("OK  devices found: %d\n", deviceCount);

    // Print device name for each GPU
    for (int d = 0; d < deviceCount; d++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, d);
        printf("OK  device %d: %s (gcnArchName: %s)\n", d, props.name, props.gcnArchName);
    }

    // Allocate, run kernel, verify output on device 0
    hipSetDevice(0);
    const int N = 64;
    float* d_out = nullptr;
    hipMalloc(&d_out, N * sizeof(float));

    hipLaunchKernelGGL(hello_gfx906, dim3(1), dim3(N), 0, 0, d_out);
    hipDeviceSynchronize();

    float h_out[N];
    hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_out);

    // Spot-check a few values
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != (float)i * 2.0f) {
            fprintf(stderr, "FAIL: h_out[%d] = %f, expected %f\n", i, h_out[i], (float)i * 2.0f);
            ok = false;
            break;
        }
    }
    if (ok) printf("OK  kernel executed and results verified\n");

    return ok ? 0 : 1;
}
