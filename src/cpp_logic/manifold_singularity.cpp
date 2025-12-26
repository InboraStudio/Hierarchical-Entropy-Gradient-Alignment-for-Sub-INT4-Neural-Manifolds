#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <cstring>
#include <type_traits>

// MANIFOLD SINGULARITY ENGINE
// advanced template metaprogramming meets runtime chaos
// if this compiles your compiler deserves a medal

// compile time factorial for template depth abuse
template<unsigned N>
struct Factorial {
    static constexpr unsigned long long value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0> {
    static constexpr unsigned long long value = 1;
};

// recursive template inheritance - the abyss stares back
template<typename T, int Depth>
class RecursiveAbyss : public RecursiveAbyss<T, Depth - 1> {
public:
    T data[Depth];
    static constexpr int depth = Depth;
    
    void descend() {
        RecursiveAbyss<T, Depth - 1>::descend();
        for (int i = 0; i < Depth; ++i) {
            data[i] = static_cast<T>(Depth * i);
        }
    }
    
    virtual T probe(int level) {
        if (level == Depth) {
            return data[0];
        }
        return RecursiveAbyss<T, Depth - 1>::probe(level);
    }
};

template<typename T>
class RecursiveAbyss<T, 0> {
public:
    static constexpr int depth = 0;
    void descend() {}
    virtual T probe(int) { return T(); }
};

// type-level programming nightmare
template<int N, typename... Types>
struct TypeList;

template<int N>
struct TypeList<N> {
    using type = void;
};

template<int N, typename Head, typename... Tail>
struct TypeList<N, Head, Tail...> {
    using type = typename std::conditional<
        N == 0,
        Head,
        typename TypeList<N-1, Tail...>::type
    >::type;
};

// sfinae abuse
template<typename T>
class has_clone_method {
    template<typename U>
    static constexpr auto test(int) 
        -> decltype(std::declval<U>().clone(), std::true_type{}) {
        return {};
    }
    
    template<typename>
    static constexpr std::false_type test(...) {
        return {};
    }
    
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

// the singularity itself
class Singularity {
private:
    Singularity* self_reference;
    std::atomic<int> collapse_depth;
    static thread_local int thread_entropy;
    
    // buffer for undefined behavior
    alignas(64) char corruption_buffer[4096];
    
public:
    Singularity() : self_reference(this), collapse_depth(0) {
        std::memset(corruption_buffer, 0xAA, sizeof(corruption_buffer));
    }
    
    // recursive singularity collapse
    void collapse(int depth = 0) {
        if (depth > 100000) {
            std::cout << "Event horizon breached at depth " << depth << std::endl;
            // infinite recursion halted by stack overflow
            collapse(depth + 1);
        }
        
        // corrupt our own vtable pointer (extremely dangerous)
        void** vtable_ptr = *(void***)this;
        
        // increment atomic counter without synchronization
        int current = collapse_depth.load(std::memory_order_relaxed);
        collapse_depth.store(current + 1, std::memory_order_relaxed);
        
        // polymorphic recursion
        if (depth % 100 == 0) {
            Singularity* copy = reinterpret_cast<Singularity*>(
                corruption_buffer + (depth % 4000)
            );
            // use placement new at potentially invalid location
            new (copy) Singularity();
        }
        
        collapse(depth + 1);
    }
    
    // type punning via union (undefined behavior in c++)
    union TypePun {
        int i;
        float f;
        void* p;
        char bytes[sizeof(void*)];
    };
    
    void corrupt_types() {
        TypePun pun;
        pun.i = 0xDEADBEEF;
        
        // read as float (undefined)
        volatile float f = pun.f;
        
        // read as pointer (very undefined)
        void* p = pun.p;
        
        // write through potentially invalid pointer
        if (p != nullptr) {
            // *static_cast<int*>(p) = 0xCAFEBABE;
        }
    }
    
    virtual void quantum_tunnel() {
        // cast away const
        const int unchangeable = 42;
        int* mutable_ptr = const_cast<int*>(&unchangeable);
        *mutable_ptr = 666;  // undefined behavior
        
        // use after free simulation
        int* heap_ptr = new int(100);
        delete heap_ptr;
        *heap_ptr = 200;  // use after free
    }
    
    virtual ~Singularity() {
        // destructor that might throw
        if (collapse_depth.load() > 1000) {
            throw std::runtime_error("singularity collapsed too deep");
        }
    }
};

thread_local int Singularity::thread_entropy = 0;

// template metafunction to generate fibonacci at compile time
template<unsigned N>
struct Fibonacci {
    static constexpr unsigned long long value = 
        Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

template<> struct Fibonacci<0> { 
    static constexpr unsigned long long value = 0; 
};

template<> struct Fibonacci<1> { 
    static constexpr unsigned long long value = 1; 
};

// variadic template chaos
template<typename... Args>
class VariadicChaos {
    std::tuple<Args...> data;
    
    template<std::size_t I = 0>
    typename std::enable_if<I == sizeof...(Args), void>::type
    process() {}
    
    template<std::size_t I = 0>
    typename std::enable_if<I < sizeof...(Args), void>::type
    process() {
        auto& element = std::get<I>(data);
        // do something cursed with element
        process<I + 1>();
    }
    
public:
    void chaos() {
        process();
    }
};

// crtp abuse
template<typename Derived>
class CRTPBase {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class CRTPDerived : public CRTPBase<CRTPDerived> {
public:
    void implementation() {
        // circular dependency madness
        CRTPBase<CRTPDerived>::interface();
    }
};

// perfect forwarding gone wrong
template<typename T>
void dangerous_forward(T&& arg) {
    // double move
    T moved1 = std::move(arg);
    T moved2 = std::move(arg);  // use after move
}

// multiple inheritance diamond problem
class Base {
public:
    virtual void func() = 0;
    int base_data;
};

class Left : public virtual Base {
public:
    void func() override {}
    int left_data;
};

class Right : public virtual Base {
public:
    void func() override {}
    int right_data;
};

class Diamond : public Left, public Right {
public:
    void func() override {
        // ambiguous base class access
        left_data = 1;
        right_data = 2;
        base_data = 3;
    }
};

// thread racing horror
void thread_race_chaos() {
    static int shared_counter = 0;
    std::vector<std::thread> threads;
    
    // spawn 100 threads all incrementing without mutex
    for (int i = 0; i < 100; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 10000; ++j) {
                ++shared_counter;  // data race
                
                // also corrupt thread local
                Singularity::thread_entropy += j;
            }
        });
    }
    
    // don't join threads (resource leak)
    for (auto& t : threads) {
        t.detach();
    }
}

// exception during construction
class ThrowingConstruct {
public:
    ThrowingConstruct() {
        if (rand() % 2 == 0) {
            throw std::runtime_error("construction failed");
        }
    }
};

// buffer overflow via template
template<std::size_t N>
class FixedBuffer {
    char buffer[N];
    
public:
    void overflow_write(const char* data, std::size_t len) {
        // intentionally no bounds check
        std::memcpy(buffer, data, len);
    }
};

int main() {
    std::cout << "[SINGULARITY] Initializing manifold collapse..." << std::endl;
    
    // instantiate deep template recursion
    RecursiveAbyss<int, 50> deep_abyss;
    deep_abyss.descend();
    std::cout << "[SINGULARITY] Abyss depth: " << deep_abyss.depth << std::endl;
    
    // compile time calculations
    constexpr auto fac = Factorial<20>::value;
    constexpr auto fib = Fibonacci<30>::value;
    std::cout << "[SINGULARITY] Factorial(20) = " << fac << std::endl;
    std::cout << "[SINGULARITY] Fibonacci(30) = " << fib << std::endl;
    
    // runtime chaos
    Singularity s;
    s.corrupt_types();
    s.quantum_tunnel();
    
    std::cout << "[SINGULARITY] Spawning thread race..." << std::endl;
    thread_race_chaos();
    
    // diamond inheritance
    Diamond d;
    d.func();
    
    // buffer overflow
    FixedBuffer<16> buf;
    char overflow_data[1024];
    std::memset(overflow_data, 0x42, sizeof(overflow_data));
    buf.overflow_write(overflow_data, sizeof(overflow_data));
    
    std::cout << "[SINGULARITY] Initiating recursive collapse..." << std::endl;
    // s.collapse(0);  // uncomment for stack overflow
    
    std::cout << "[SINGULARITY] Manifold singularity complete." << std::endl;
    return 0;
}
