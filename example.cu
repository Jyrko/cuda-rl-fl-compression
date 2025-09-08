// cuda_core_principles.cu
// nvcc -O2 -arch=native cuda_core_principles.cu -o cuda_demo
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::exit(1); \
  } \
} while(0)

// ---------- Constant memory (cached, read-only from kernels) ----------
__constant__ float cAlpha;

// ---------- 1) Basic global memory kernel (vector add) ----------
__global__ void vecAdd(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c,
                       int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 1D indexing
  if (i < n) c[i] = a[i] + b[i];
}

// ---------- 2) SAXPY using constant memory ----------
__global__ void saxpyConst(const float* __restrict__ x,
                           const float* __restrict__ y,
                           float* __restrict__ out,
                           int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cAlpha * x[i] + y[i];
}

// ---------- 3) Block-level reduction using shared memory + syncthreads ----------
template<int BLOCK_SIZE>
__global__ void reduceSumShared(const float* __restrict__ x,
                                float* __restrict__ blockSums,
                                int n)
{
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int i   = blockIdx.x * BLOCK_SIZE * 2 + tid; // two elements per thread (load coalescing)
  float val = 0.f;
  if (i < n)              val += x[i];
  if (i + BLOCK_SIZE < n) val += x[i + BLOCK_SIZE];
  sdata[tid] = val;
  __syncthreads();

  // tree reduce within the block
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) blockSums[blockIdx.x] = sdata[0];
}

// ---------- 4) Atomics example: histogram into 256 bins ----------
__global__ void histogram256(const unsigned char* data, int n, unsigned int* bins256)
{
  // Optionally: use shared memory per block to reduce contention, then merge.
  __shared__ unsigned int sBins[256];
  int t = threadIdx.x;
  // init shared bins
  for (int k = t; k < 256; k += blockDim.x) sBins[k] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + t;
  if (i < n) atomicAdd(&sBins[data[i]], 1);
  __syncthreads();

  // merge to global memory
  for (int k = t; k < 256; k += blockDim.x)
    atomicAdd(&bins256[k], sBins[k]);
}

int main() {
  // ---------------- Problem size & launch config ----------------
  const int N = 1 << 20;                // 1,048,576 elements
  const int BLOCK = 256;
  const int GRID  = (N + BLOCK - 1) / BLOCK;

  // ---------------- Host data ----------------
  std::vector<float> hA(N), hB(N), hC(N), hY(N), hOut(N);
  std::iota(hA.begin(), hA.end(), 0.f);
  std::fill(hB.begin(), hB.end(), 1.0f);
  std::fill(hY.begin(), hY.end(), 2.0f);

  // Histogram host data (bytes)
  std::vector<unsigned char> hBytes(N);
  for (int i = 0; i < N; ++i) hBytes[i] = static_cast<unsigned char>(i % 256);
  std::vector<unsigned int> hHist(256, 0);

  // ---------------- Device allocations ----------------
  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dY = nullptr, *dOut = nullptr;
  unsigned char *dBytes = nullptr;
  unsigned int *dBins = nullptr;

  CHECK_CUDA(cudaMalloc(&dA, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dB, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dC, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dY, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dBytes, N * sizeof(unsigned char)));
  CHECK_CUDA(cudaMalloc(&dBins, 256 * sizeof(unsigned int)));
  CHECK_CUDA(cudaMemset(dBins, 0, 256 * sizeof(unsigned int)));

  // ---------------- Streams to overlap copies/compute ----------------
  cudaStream_t s1, s2;
  CHECK_CUDA(cudaStreamCreate(&s1));
  CHECK_CUDA(cudaStreamCreate(&s2));

  // Split the workload in halves for demonstration
  const int N2 = N / 2;
  const size_t bytesF = N2 * sizeof(float);
  const size_t bytesB = N2 * sizeof(unsigned char);

  // Async H2D copies on different streams
  CHECK_CUDA(cudaMemcpyAsync(dA, hA.data(), bytesF, cudaMemcpyHostToDevice, s1));
  CHECK_CUDA(cudaMemcpyAsync(dA + N2, hA.data() + N2, bytesF, cudaMemcpyHostToDevice, s2));

  CHECK_CUDA(cudaMemcpyAsync(dB, hB.data(), bytesF, cudaMemcpyHostToDevice, s1));
  CHECK_CUDA(cudaMemcpyAsync(dB + N2, hB.data() + N2, bytesF, cudaMemcpyHostToDevice, s2));

  CHECK_CUDA(cudaMemcpyAsync(dY, hY.data(), bytesF, cudaMemcpyHostToDevice, s1));
  CHECK_CUDA(cudaMemcpyAsync(dY + N2, hY.data() + N2, bytesF, cudaMemcpyHostToDevice, s2));

  CHECK_CUDA(cudaMemcpyAsync(dBytes, hBytes.data(), bytesB, cudaMemcpyHostToDevice, s1));
  CHECK_CUDA(cudaMemcpyAsync(dBytes + N2, hBytes.data() + N2, bytesB, cudaMemcpyHostToDevice, s2));

  // ---------------- Timing with events ----------------
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  // Set constant memory for saxpy
  float alpha = 2.5f;
  CHECK_CUDA(cudaMemcpyToSymbol(cAlpha, &alpha, sizeof(float)));

  // Launch on each half on separate streams
  int gridHalf = (N2 + BLOCK - 1) / BLOCK;
  vecAdd<<<gridHalf, BLOCK, 0, s1>>>(dA, dB, dC, N2);
  vecAdd<<<gridHalf, BLOCK, 0, s2>>>(dA + N2, dB + N2, dC + N2, N2);
  // Chain SAXPY (out = alpha*a + y)
  saxpyConst<<<gridHalf, BLOCK, 0, s1>>>(dA, dY, dOut, N2);
  saxpyConst<<<gridHalf, BLOCK, 0, s2>>>(dA + N2, dY + N2, dOut + N2, N2);
  // Histogram with atomics
  histogram256<<<gridHalf, BLOCK, 0, s1>>>(dBytes, N2, dBins);
  histogram256<<<gridHalf, BLOCK, 0, s2>>>(dBytes + N2, N2, dBins);

  // Reduction uses single stream (for simplicity)
  // Step 1: block partials
  int gridRed = (N + (BLOCK * 2 - 1)) / (BLOCK * 2);
  float* dBlockSums = nullptr;
  CHECK_CUDA(cudaMalloc(&dBlockSums, gridRed * sizeof(float)));
  reduceSumShared<BLOCK><<<gridRed, BLOCK>>>(dOut, dBlockSums, N);
  // Step 2: final reduction on CPU (small)
  std::vector<float> hBlockSums(gridRed);
  CHECK_CUDA(cudaMemcpy(hBlockSums.data(), dBlockSums, gridRed * sizeof(float), cudaMemcpyDeviceToHost));
  float sumOut = std::accumulate(hBlockSums.begin(), hBlockSums.end(), 0.0f);

  // Wait for async work and time
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  // ---------------- D2H copies ----------------
  CHECK_CUDA(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hHist.data(), dBins, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  // ---------------- Verify a few things ----------------
  bool okAdd = (hC[123] == hA[123] + hB[123]) && (hC[N-1] == hA[N-1] + hB[N-1]);
  bool okSax = (std::abs(hOut[777] - (alpha * hA[777] + hY[777])) < 1e-5f);
  unsigned int histSum = std::accumulate(hHist.begin(), hHist.end(), 0u);

  // Device info
  cudaDeviceProp prop{};
  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

  printf("Device: %s | SMs: %d | MaxThreads/Block: %d\n", prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock);
  printf("vecAdd correct: %s | saxpy correct: %s | histogram count: %u (expected %d)\n",
         okAdd ? "yes" : "NO", okSax ? "yes" : "NO", histSum, N);
  printf("reduce(sum(out)) ≈ %.3f  (first 5 out: %.2f %.2f %.2f %.2f %.2f)\n",
         sumOut, hOut[0], hOut[1], hOut[2], hOut[3], hOut[4]);
  printf("Elapsed: %.3f ms (copies+compute overlapped via 2 streams)\n", ms);

  // ---------------- Cleanup ----------------
  CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaFree(dY)); CHECK_CUDA(cudaFree(dOut));
  CHECK_CUDA(cudaFree(dBytes)); CHECK_CUDA(cudaFree(dBins));
  CHECK_CUDA(cudaFree(dBlockSums));
  CHECK_CUDA(cudaStreamDestroy(s1)); CHECK_CUDA(cudaStreamDestroy(s2));
  CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
  return 0;
}
