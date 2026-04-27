// sequential_bucket_sort.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

// Bucket sort - дараалсан хувилбар
void bucketSort(std::vector<float>& arr) {
    int n = arr.size();
    if (n <= 0) return;

    // n ширхэг хувин үүсгэх
    std::vector<std::vector<float>> buckets(n);

    // Элементүүдийг хувинд хуваарилах [0, 1) мужид
    for (int i = 0; i < n; i++) {
        int idx = (int)(n * arr[i]); // Хувин индекс тооцох
        if (idx >= n) idx = n - 1;   // Хил давахаас сэргийлэх
        buckets[idx].push_back(arr[i]);
    }

    // Хувин бүрийг дотооддоо эрэмбэлэх
    for (int i = 0; i < n; i++) {
        std::sort(buckets[i].begin(), buckets[i].end());
    }

    // Хувинуудыг нэгтгэж эцсийн массив үүсгэх
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (float val : buckets[i]) {
            arr[idx++] = val;
        }
    }
}

int main() {
    std::vector<int> sizes = {10000, 100000, 1000000};

    for (int n : sizes) {
        // [0, 1) мужид санамсаргүй тоо үүсгэх
        std::vector<float> arr(n);
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : arr) x = dist(rng);

        auto start = std::chrono::high_resolution_clock::now();
        bucketSort(arr);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Sequential | N=" << n << " | Time=" << ms << " ms\n";
    }
    return 0;
}

//++ -O2 -std=c++17 seq_sort.cpp -o seq_sort
//./seq_sort