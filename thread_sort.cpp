// thread_bucket_sort.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>

// Хувин бүрийг тогтоосон мужид эрэмбэлэх thread функц
void sortBucketRange(std::vector<std::vector<float>>& buckets,
                     int start, int end) {
    for (int i = start; i < end; i++) {
        std::sort(buckets[i].begin(), buckets[i].end());
    }
}

void parallelBucketSort(std::vector<float>& arr) {
    int n = arr.size();
    int numBuckets = 1024;

    // Хувинуудыг үүсгэх
    std::vector<std::vector<float>> buckets(numBuckets);

    // Элементүүдийг хувинд хуваарилах
    for (int i = 0; i < n; i++) {
        int idx = (int)(arr[i] * numBuckets);
        if (idx >= numBuckets) idx = numBuckets - 1;
        buckets[idx].push_back(arr[i]);
    }

    // Боломжит thread тоог авах
    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads(numThreads);

    // Хувинуудыг thread-үүдэд тэнцүү хуваарилах
    int bucketsPerThread = numBuckets / numThreads;

    for (unsigned int t = 0; t < numThreads; t++) {
        int startB = t * bucketsPerThread;
        int endB   = (t == numThreads - 1) ? numBuckets
                                           : startB + bucketsPerThread;
        threads[t] = std::thread(sortBucketRange,
                                 std::ref(buckets),
                                 startB, endB);
    }

    // Бүх thread дуусахыг хүлээх
    for (auto& th : threads) th.join();

    // Нэгтгэх
    int idx = 0;
    for (auto& b : buckets)
        for (float v : b)
            arr[idx++] = v;
}

bool verifySorted(const std::vector<float>& arr) {
    for (int i = 1; i < (int)arr.size(); i++)
        if (arr[i] < arr[i-1]) return false;
    return true;
}

int main() {
    unsigned int nt = std::thread::hardware_concurrency();
    std::cout << "Available threads: " << nt << "\n\n";

    for (int n : {10000, 100000, 1000000}) {
        std::vector<float> arr(n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : arr) x = dist(rng);

        auto t1 = std::chrono::high_resolution_clock::now();
        parallelBucketSort(arr);
        auto t2 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double,
                    std::milli>(t2 - t1).count();

        std::cout << "std::thread | N=" << n
                  << " | Time=" << ms << " ms"
                  << " | Sorted: "
                  << (verifySorted(arr) ? "YES ✓" : "NO ✗")
                  << "\n";
    }
    return 0;
}

//g++ -O2 -std=c++17 thread_sort.cpp -pthread -o thread_sort && echo "✓ thread_sort compiled"
//./thread_sort