// cuda_bucket_sort.cu
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// ============================================================
// CUDA Kernel 1: Элемент бүрийг хувинд хуваарилах
// Гаралт: bucketIds[i] = i-р элементийн хувин дугаар
// ============================================================
__global__ void assignBuckets(
    const float* arr,
    int* bucketIds,
    int n,
    int numBuckets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // [0,1) утгыг хувин индекс болгох
        int idx = (int)(arr[tid] * numBuckets);
        // Хил давахаас сэргийлэх (arr[i] == 1.0 үед)
        if (idx >= numBuckets) idx = numBuckets - 1;
        bucketIds[tid] = idx;
    }
}

// ============================================================
// CUDA Kernel 2: Atomic ашиглан хувин бүрийн элемент тоолох
// ============================================================
__global__ void countBuckets(
    const int* bucketIds,
    int* bucketCounts,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Race condition-аас сэргийлж atomicAdd ашиглана
        atomicAdd(&bucketCounts[bucketIds[tid]], 1);
    }
}

// ============================================================
// CUDA Kernel 3: Элементүүдийг хувин бүрт байрлуулах
// bucketOffsets ашиглан зөв байрлалд бичнэ
// ============================================================
__global__ void scatterToBuckets(
    const float* arr,
    const int* bucketIds,
    float* output,
    int* bucketOffsets,    // Хувин бүрийн эхлэх байрлал
    int* bucketPositions,  // Атомик тоолуур (thread-safe)
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int bucket = bucketIds[tid];
        // Энэ хувинд хэдэн дэх элемент болохыг атомикаар авах
        int pos = atomicAdd(&bucketPositions[bucket], 1);
        output[bucketOffsets[bucket] + pos] = arr[tid];
    }
}

// ============================================================
// Host функц: Prefix Sum (CPU дээр - жижиг массивт хурдан)
// bucketCounts -> bucketOffsets
// ============================================================
void exclusivePrefixSum(
    const std::vector<int>& counts,
    std::vector<int>& offsets)
{
    offsets[0] = 0;
    for (int i = 1; i < (int)counts.size(); i++) {
        offsets[i] = offsets[i-1] + counts[i-1];
    }
}

// ============================================================
// Үндсэн CUDA Bucket Sort функц
// ============================================================
void cudaBucketSort(std::vector<float>& arr) {
    int n = (int)arr.size();
    // Хувин тоо - GPU thread block-тай нийцүүлэх
    int numBuckets = 1024;

    // ---------- CUDA хугацаа хэмжигч ----------
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // ===== ШАТЛАЛ 1: Санах ой зарлах =====
    float *d_arr, *d_output;
    int *d_bucketIds, *d_bucketCounts, *d_bucketPositions, *d_bucketOffsets;

    cudaMalloc(&d_arr,             n * sizeof(float));
    cudaMalloc(&d_output,          n * sizeof(float));
    cudaMalloc(&d_bucketIds,       n * sizeof(int));
    cudaMalloc(&d_bucketCounts,    numBuckets * sizeof(int));
    cudaMalloc(&d_bucketPositions, numBuckets * sizeof(int));
    cudaMalloc(&d_bucketOffsets,   numBuckets * sizeof(int));

    // ===== ШАТЛАЛ 2: CPU -> GPU өгөгдөл дамжуулах =====
    cudaEventRecord(startEvent);

    cudaMemcpy(d_arr, arr.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_bucketCounts,    0, numBuckets * sizeof(int));
    cudaMemset(d_bucketPositions, 0, numBuckets * sizeof(int));

    float transferTime = 0;
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&transferTime, startEvent, stopEvent);
    std::cout << "  [Data Transfer H->D] " << transferTime << " ms\n";

    // ===== ШАТЛАЛ 3: Kernel тохиргоо =====
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    cudaEvent_t k1, k2;
    cudaEventCreate(&k1);
    cudaEventCreate(&k2);
    cudaEventRecord(k1);

    // Kernel 1: Хувин индекс тооцох
    assignBuckets<<<gridSize, blockSize>>>(d_arr, d_bucketIds, n, numBuckets);

    // Kernel 2: Хувин бүрийн элемент тоолох
    countBuckets<<<gridSize, blockSize>>>(d_bucketIds, d_bucketCounts, n);
    cudaDeviceSynchronize();

    cudaEventRecord(k2);
    cudaEventSynchronize(k2);
    float kernelTime = 0;
    cudaEventElapsedTime(&kernelTime, k1, k2);
    std::cout << "  [Kernel 1+2 Time]     " << kernelTime << " ms\n";

    // ===== ШАТЛАЛ 4: CPU дээр Prefix Sum тооцох =====
    std::vector<int> h_bucketCounts(numBuckets);
    std::vector<int> h_bucketOffsets(numBuckets);

    cudaMemcpy(h_bucketCounts.data(), d_bucketCounts,
               numBuckets * sizeof(int), cudaMemcpyDeviceToHost);

    exclusivePrefixSum(h_bucketCounts, h_bucketOffsets);

    // Offset-уудыг GPU руу буцаах
    cudaMemcpy(d_bucketOffsets, h_bucketOffsets.data(),
               numBuckets * sizeof(int), cudaMemcpyHostToDevice);

    // ===== ШАТЛАЛ 5: Элементүүдийг хувинд байрлуулах =====
    scatterToBuckets<<<gridSize, blockSize>>>(
        d_arr, d_bucketIds, d_output,
        d_bucketOffsets, d_bucketPositions, n);
    cudaDeviceSynchronize();

    // ===== ШАТЛАЛ 6: Хувин бүрийг Thrust-аар эрэмбэлэх =====
    cudaEvent_t s1, s2;
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);
    cudaEventRecord(s1);

    thrust::device_ptr<float> d_out_ptr(d_output);
    for (int b = 0; b < numBuckets; b++) {
        if (h_bucketCounts[b] > 1) {
            // Хувин бүрийн өөрийн хэсгийг тусад нь эрэмбэлэх
            thrust::sort(
                thrust::device,
                d_out_ptr + h_bucketOffsets[b],
                d_out_ptr + h_bucketOffsets[b] + h_bucketCounts[b]
            );
        }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(s2);
    cudaEventSynchronize(s2);
    float sortTime = 0;
    cudaEventElapsedTime(&sortTime, s1, s2);
    std::cout << "  [Bucket Sort Time]    " << sortTime << " ms\n";

    // ===== ШАТЛАЛ 7: GPU -> CPU үр дүн буцаах =====
    cudaEvent_t d1, d2;
    cudaEventCreate(&d1);
    cudaEventCreate(&d2);
    cudaEventRecord(d1);

    cudaMemcpy(arr.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(d2);
    cudaEventSynchronize(d2);
    float downloadTime = 0;
    cudaEventElapsedTime(&downloadTime, d1, d2);
    std::cout << "  [Data Transfer D->H]  " << downloadTime << " ms\n";

    // ===== ШАТЛАЛ 8: Санах ой чөлөөлөх =====
    cudaFree(d_arr);
    cudaFree(d_output);
    cudaFree(d_bucketIds);
    cudaFree(d_bucketCounts);
    cudaFree(d_bucketPositions);
    cudaFree(d_bucketOffsets);

    // Event устгах
    cudaEventDestroy(startEvent); cudaEventDestroy(stopEvent);
    cudaEventDestroy(k1);  cudaEventDestroy(k2);
    cudaEventDestroy(s1);  cudaEventDestroy(s2);
    cudaEventDestroy(d1);  cudaEventDestroy(d2);
}

// ============================================================
// Үр дүн шалгах функц
// ============================================================
bool verifySorted(const std::vector<float>& arr) {
    for (int i = 1; i < (int)arr.size(); i++) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

int main() {
    std::vector<int> sizes = {10000, 100000, 1000000};

    for (int n : sizes) {
        std::cout << "\n===== N = " << n << " =====\n";

        std::vector<float> arr(n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : arr) x = dist(rng);

        // Нийт гүйцэтгэлийн хугацаа (дамжуулалт + compute)
        cudaEvent_t t1, t2;
        cudaEventCreate(&t1);
        cudaEventCreate(&t2);
        cudaEventRecord(t1);

        cudaBucketSort(arr);

        cudaEventRecord(t2);
        cudaEventSynchronize(t2);
        float totalMs = 0;
        cudaEventElapsedTime(&totalMs, t1, t2);

        std::cout << "  [Total CUDA Time]     " << totalMs << " ms\n";
        std::cout << "  Sorted correctly: "
                  << (verifySorted(arr) ? "YES ✓" : "NO ✗") << "\n";

        cudaEventDestroy(t1);
        cudaEventDestroy(t2);
    }
    return 0;
}