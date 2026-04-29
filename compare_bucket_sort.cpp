#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <omp.h>
using namespace std;
using Clock = chrono::high_resolution_clock;
using ms = chrono::duration<double, milli>;
void seqBucketSort(vector<float>& arr) {
    int n=(int)arr.size(),K=1024;
    vector<vector<float>> buckets(K);
    for(int i=0;i<n;i++){int idx=(int)(arr[i]*K);if(idx>=K)idx=K-1;buckets[idx].push_back(arr[i]);}
    for(auto& b:buckets)sort(b.begin(),b.end());
    int idx=0;for(auto& b:buckets)for(float v:b)arr[idx++]=v;
}
void sortRange(vector<vector<float>>& buckets,int s,int e){for(int i=s;i<e;i++)sort(buckets[i].begin(),buckets[i].end());}
void threadBucketSort(vector<float>& arr){
    int n=(int)arr.size(),K=1024;
    vector<vector<float>> buckets(K);
    for(int i=0;i<n;i++){int idx=(int)(arr[i]*K);if(idx>=K)idx=K-1;buckets[idx].push_back(arr[i]);}
    unsigned int nt=thread::hardware_concurrency();
    vector<thread> threads(nt);int per=K/nt;
    for(unsigned int t=0;t<nt;t++){int s=t*per,e=(t==nt-1)?K:s+per;threads[t]=thread(sortRange,ref(buckets),s,e);}
    for(auto& th:threads)th.join();
    int idx=0;for(auto& b:buckets)for(float v:b)arr[idx++]=v;
}
void ompBucketSort(vector<float>& arr){
    int n=(int)arr.size(),K=1024;
    vector<vector<float>> buckets(K);
    for(int i=0;i<n;i++){int idx=(int)(arr[i]*K);if(idx>=K)idx=K-1;buckets[idx].push_back(arr[i]);}
    #pragma omp parallel for schedule(dynamic) num_threads(8)
    for(int i=0;i<K;i++)if(!buckets[i].empty())sort(buckets[i].begin(),buckets[i].end());
    int idx=0;for(auto& b:buckets)for(float v:b)arr[idx++]=v;
}
vector<float> makeData(int n){vector<float> arr(n);mt19937 rng(42);uniform_real_distribution<float> dist(0.f,1.f);for(auto& x:arr)x=dist(rng);return arr;}
bool isSorted(const vector<float>& arr){for(int i=1;i<(int)arr.size();i++)if(arr[i]<arr[i-1])return false;return true;}
double totalOps(int n,int K=1024){return (double)n+((double)n*log2((double)n/K))+(double)n;}
double dataTransferBytes(int n){return 4.0*n*sizeof(float);}
struct Result{double execMs;bool ok;};
Result bench(int n,void(*fn)(vector<float>&),int reps=3){
    double total=0;bool sorted=true;
    for(int r=0;r<reps;r++){auto data=makeData(n);auto t1=Clock::now();fn(data);auto t2=Clock::now();total+=ms(t2-t1).count();if(!isSorted(data))sorted=false;}
    return{total/reps,sorted};
}
int main(){
    cout<<"CPU threads: "<<thread::hardware_concurrency()<<"\n";
    cout<<"OMP threads: "<<omp_get_max_threads()<<"\n\n";
    cout<<left<<setw(10)<<"N"<<setw(12)<<"Method"<<setw(12)<<"Exec(ms)"<<setw(10)<<"SpeedUp"<<setw(14)<<"Ops(M)"<<setw(14)<<"MB"<<setw(14)<<"MOPS"<<setw(6)<<"OK?"<<"\n"<<string(92,'-')<<"\n";
    for(int n:{10000,100000,1000000}){
        auto rS=bench(n,seqBucketSort),rT=bench(n,threadBucketSort),rO=bench(n,ompBucketSort);
        double opsM=totalOps(n)/1e6,xMB=dataTransferBytes(n)/1e6;
        auto row=[&](string lab,Result r,double su){
            cout<<left<<setw(10)<<(lab=="Sequential"?to_string(n):"")<<setw(12)<<lab<<setw(12)<<fixed<<setprecision(3)<<r.execMs<<setw(10)<<setprecision(2)<<su<<setw(14)<<setprecision(2)<<opsM<<setw(14)<<setprecision(3)<<xMB<<setw(14)<<setprecision(1)<<opsM/(r.execMs/1000.0)<<setw(6)<<(r.ok?"YES":"NO")<<"\n";
        };
        row("Sequential",rS,1.0);row("std::thread",rT,rS.execMs/rT.execMs);row("OpenMP",rO,rS.execMs/rO.execMs);
        cout<<string(92,'-')<<"\n";
    }
    return 0;
}
