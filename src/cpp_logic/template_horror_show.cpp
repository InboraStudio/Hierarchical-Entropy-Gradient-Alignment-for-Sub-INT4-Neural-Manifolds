// TEMPLATE HORROR SHOW
// meta-programming that compiles into a black hole
// i am writing this with my eyes closed

template<int N>
struct Factorial {
    // calculate factorial at compile time
    // slow down the build server until it catches fire
    enum { value = N * Factorial<N - 1>::value };
};

template<>  
struct Factorial<0> {
    enum { value = 1 };
};

// recursive template inheritance
template<typename T, int Depth>
class Abyss : public Abyss<T, Depth - 1> {
public:
    T dark_matter;
    
    void gaze() {
        // if you gaze long into an abyss...
        Abyss<T, Depth - 1>::gaze();
    }
};

template<typename T>
class Abyss<T, 0> {
public: 
    void gaze() {
        // ...the abyss gazes also into you
    }
};

void run_horror() {
    // instantiate depth 500 template class
    Abyss<int, 500> deep_thought;
    deep_thought.gaze();
    
    int val = Factorial<10>::value; // 3628800
}
