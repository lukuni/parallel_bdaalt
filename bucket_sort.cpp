
#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <omp.h>

using namespace std;
using Clock = chrono::high_resolution_clock;using ms    = chrono::duration<double, milli>;

// ─────────────────────────────────────────────
//  SEQUENTIAL
// ─────────────────────────────────────────────
void seqBucketSort(vector<float>& arr) {
    int n = (int)arr.size();
    int K = 1024;
    vector<vector<float>> buckets(K);

    // Scatter
    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * K);
        if (idx >= K) idx = K - 1;
        buckets[idx].push_back(arr[i]);
    }
    // Sort
    for (auto& b : buckets) sort(b.begin(), b.end());
    // Gather
    int idx = 0;
    for (auto& b : buckets)
        for (float v : b) arr[idx++] = v;
}

// ─────────────────────────────────────────────
//  STD::THREAD  (parallel sort phase)
// ─────────────────────────────────────────────
void sortRange(vector<vector<float>>& buckets, int s, int e) {
    for (int i = s; i < e; i++)
        sort(buckets[i].begin(), buckets[i].end());
}

void threadBucketSort(vector<float>& arr) {
    int n = (int)arr.size();
    int K = 1024;
    vector<vector<float>> buckets(K);

    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * K);
        if (idx >= K) idx = K - 1;
        buckets[idx].push_back(arr[i]);
    }

    unsigned int nt = thread::hardware_concurrency();
    vector<thread> threads(nt);
    int per = K / nt;
    for (unsigned int t = 0; t < nt; t++) {
        int s = t * per;
        int e = (t == nt - 1) ? K : s + per;
        threads[t] = thread(sortRange, ref(buckets), s, e);
    }
    for (auto& th : threads) th.join();

    int idx = 0;
    for (auto& b : buckets)
        for (float v : b) arr[idx++] = v;
}

// ─────────────────────────────────────────────
//  OPENMP  (parallel sort phase, dynamic schedule)
// ─────────────────────────────────────────────
void ompBucketSort(vector<float>& arr) {
    int n = (int)arr.size();
    int K = 1024;
    vector<vector<float>> buckets(K);

    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * K);
        if (idx >= K) idx = K - 1;
        buckets[idx].push_back(arr[i]);
    }

    #pragma omp parallel for schedule(dynamic) num_threads(8)
    for (int i = 0; i < K; i++)
        if (!buckets[i].empty())
            sort(buckets[i].begin(), buckets[i].end());

    int idx = 0;
    for (auto& b : buckets)
        for (float v : b) arr[idx++] = v;
}

// ─────────────────────────────────────────────
//  HELPERS
// ─────────────────────────────────────────────
vector<float> makeData(int n) {
    vector<float> arr(n);
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& x : arr) x = dist(rng);
    return arr;
}

bool isSorted(const vector<float>& arr) {
    for (int i = 1; i < (int)arr.size(); i++)
        if (arr[i] < arr[i-1]) return false;
    return true;
}

// Estimated total operations:
//   Scatter:  n  (one index compute per element)
//   Sort:     K * (n/K)*log2(n/K)  = n*log2(n/K)
//   Gather:   n
double totalOps(int n, int K = 1024) {
    double sortOps = (double)n * log2((double)n / K);
    return (double)n + sortOps + (double)n;   // scatter + sort + gather
}

// Data transferred (bytes):
//   The input array is read once (Scatter) and written once (Gather) = 2*n*4 bytes
//   Buckets are written during Scatter and read during Sort+Gather ~ 2*n*4
//   Total ~ 4 * n * sizeof(float)
double dataTransferBytes(int n) {
    return 4.0 * n * sizeof(float);
}

struct Result {
    double execMs;   // wall time
    double compMs;   // computation time (excludes data generation)
    bool   ok;
};

Result bench(int n, void(*fn)(vector<float>&), int reps = 3) {
    double totalExec = 0;
    bool sorted = true;
    for (int r = 0; r < reps; r++) {
        auto data = makeData(n);
        auto t1   = Clock::now();
        fn(data);
        auto t2   = Clock::now();
        totalExec += ms(t2 - t1).count();
        if (!isSorted(data)) sorted = false;
    }
    double avg = totalExec / reps;
    return {avg, avg, sorted};   // compMs ≈ execMs (no I/O here)
}

// ─────────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────────
int main() {
    int cpuThreads = thread::hardware_concurrency();
    int ompThreads = omp_get_max_threads();

    cout << "========================================================\n";
    cout << "  Bucket Sort — Sequential vs std::thread vs OpenMP\n";
    cout << "========================================================\n";
    cout << "CPU logical threads : " << cpuThreads << "\n";
    cout << "OMP max threads     : " << ompThreads << "\n";
    cout << "Bucket count (K)    : 1024\n";
    cout << "Runs per config     : 3  (averaged)\n\n";

    // ── Table header ──────────────────────────────────────────────────────────
    cout << left
         << setw(10) << "N"
         << setw(12) << "Method"
         << setw(14) << "Exec(ms)"
         << setw(14) << "Comp(ms)"
         << setw(14) << "SpeedUp"
         << setw(18) << "TotalOps(M)"
         << setw(20) << "DataXfer(MB)"
         << setw(20) << "Perf(MOPS)"
         << setw(8)  << "OK?"
         << "\n"
         << string(130, '-') << "\n";

    for (int n : {10000, 100000, 1000000}) {
        auto rSeq    = bench(n, seqBucketSort);
        auto rThread = bench(n, threadBucketSort);
        auto rOmp    = bench(n, ompBucketSort);

        double ops    = totalOps(n);           // raw ops
        double opsM   = ops / 1e6;             // MegaOps
        double xferMB = dataTransferBytes(n) / 1e6;

        // Achievable performance = TotalOps / ExecutionTime
        double perfSeq    = opsM / (rSeq.execMs    / 1000.0);
        double perfThread = opsM / (rThread.execMs / 1000.0);
        double perfOmp    = opsM / (rOmp.execMs    / 1000.0);

        double suThread = rSeq.execMs / rThread.execMs;
        double suOmp    = rSeq.execMs / rOmp.execMs;

        auto row = [&](const string& label, const Result& r,
                       double speedup, double perf) {
            cout << left
                 << setw(10) << (label == "Sequential" ? to_string(n) : "")
                 << setw(12) << label
                 << setw(14) << fixed << setprecision(3) << r.execMs
                 << setw(14) << r.compMs
                 << setw(14) << setprecision(2) << speedup
                 << setw(18) << setprecision(2) << opsM
                 << setw(20) << setprecision(3) << xferMB
                 << setw(20) << setprecision(1) << perf
                 << setw(8)  << (r.ok ? "YES✓" : "NO✗")
                 << "\n";
        };

        row("Sequential", rSeq,    1.00,     perfSeq);
        row("std::thread", rThread, suThread, perfThread);
        row("OpenMP",      rOmp,    suOmp,    perfOmp);
        cout << string(130, '-') << "\n";
    }

    // ── Summary legend ────────────────────────────────────────────────────────
    cout << "\nLEGEND\n";
    cout << "  Exec(ms)     : Total wall-clock time (avg of 3 runs)\n";
    cout << "  Comp(ms)     : Computation time (= Exec here; no disk I/O)\n";
    cout << "  SpeedUp      : Sequential_time / This_method_time\n";
    cout << "  TotalOps(M)  : n + n*log2(n/K) + n  (scatter+sort+gather)\n";
    cout << "  DataXfer(MB) : 4 * n * sizeof(float)  (RAM read+write)\n";
    cout << "  Perf(MOPS)   : TotalOps(M) / Exec(s)  — achievable throughput\n";

    return 0;
}

