#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>

using namespace std;
using ms = chrono::duration<double, milli>;

void seqBucketSort(vector<float>& arr) {
    int n = arr.size();
    int numBuckets = 1024;
    vector<vector<float>> buckets(numBuckets);
    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * numBuckets);
        if (idx >= numBuckets) idx = numBuckets - 1;
        buckets[idx].push_back(arr[i]);
    }
    for (auto& b : buckets) sort(b.begin(), b.end());
    int idx = 0;
    for (auto& b : buckets)
        for (float v : b) arr[idx++] = v;
}

void sortRange(vector<vector<float>>& buckets, int start, int end) {
    for (int i = start; i < end; i++)
        sort(buckets[i].begin(), buckets[i].end());
}

void threadBucketSort(vector<float>& arr) {
    int n = arr.size();
    int numBuckets = 1024;
    vector<vector<float>> buckets(numBuckets);
    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * numBuckets);
        if (idx >= numBuckets) idx = numBuckets - 1;
        buckets[idx].push_back(arr[i]);
    }
    unsigned int nt = thread::hardware_concurrency();
    vector<thread> threads(nt);
    int perThread = numBuckets / nt;
    for (unsigned int t = 0; t < nt; t++) {
        int s = t * perThread;
        int e = (t == nt - 1) ? numBuckets : s + perThread;
        threads[t] = thread(sortRange, ref(buckets), s, e);
    }
    for (auto& th : threads) th.join();
    int idx = 0;
    for (auto& b : buckets)
        for (float v : b) arr[idx++] = v;
}

void ompBucketSort(vector<float>& arr) {
    int n = arr.size();
    int numBuckets = 1024;
    vector<vector<float>> buckets(numBuckets);
    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * numBuckets);
        if (idx >= numBuckets) idx = numBuckets - 1;
        buckets[idx].push_back(arr[i]);
    }
    #pragma omp parallel for schedule(dynamic) num_threads(8)
    for (int i = 0; i < numBuckets; i++)
        if (!buckets[i].empty())
            sort(buckets[i].begin(), buckets[i].end());
    int idx = 0;
    for (auto& b : buckets)
        for (float v : b) arr[idx++] = v;
}

vector<float> generateData(int n) {
    vector<float> arr(n);
    mt19937 rng(42);
    uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : arr) x = dist(rng);
    return arr;
}

// Warmup + 10 удаа хэмжиж дундаж авах
double measureTime(int n, void(*fn)(vector<float>&)) {
    // 3 удаа warmup - cache, thread pool дулаацуулах
    for (int w = 0; w < 3; w++) {
        auto data = generateData(n);
        fn(data);
    }

    // 10 удаа хэмжих
    double total = 0;
    for (int r = 0; r < 10; r++) {
        auto data = generateData(n);
        auto t1 = chrono::high_resolution_clock::now();
        fn(data);
        auto t2 = chrono::high_resolution_clock::now();
        total += ms(t2 - t1).count();
    }
    return total / 10.0; // Дундаж
}

int main() {
    cout << "CPU threads : " << thread::hardware_concurrency() << "\n";
    cout << "OMP threads : " << omp_get_max_threads() << "\n";
    cout << "Warmup runs : 10\n";
    cout << "Measure runs: 10\n\n";

    cout << left
         << setw(10) << "N"
         << setw(16) << "Seq(ms)"
         << setw(16) << "Thread(ms)"
         << setw(16) << "OMP(ms)"
         << setw(16) << "SpeedUp(T)"
         << setw(16) << "SpeedUp(OMP)"
         << "\n" << string(90, '-') << "\n";

    for (int n : {10000, 100000, 1000000}) {
        cout << "  Running N=" << n << "..." << flush;

        double tSeq    = measureTime(n, seqBucketSort);
        double tThread = measureTime(n, threadBucketSort);
        double tOmp    = measureTime(n, ompBucketSort);

        cout << "\r" << left
             << setw(10) << n
             << setw(16) << fixed << setprecision(3) << tSeq
             << setw(16) << tThread
             << setw(16) << tOmp
             << setw(16) << setprecision(2) << tSeq/tThread
             << setw(16) << tSeq/tOmp
             << "\n";
    }

    cout << "\n(Дундаж 10 удаагийн хэмжилт, warmup 3)\n";
    return 0;
}