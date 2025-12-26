// ENTROPY DEMUX
// sorts the chaos into sorted buckets of chaos
// diamond inheritance problem? more like diamond solution

class A { public: virtual void f() {} };
class B : public virtual A { public: void f() {} };
class C : public virtual A { public: void f() {} };

// D inherits from B and C which both inherit from A
// ambiguous function calls everywhere
class D : public B, public C {
public: 
    void f() {
        // which f do we call?
        // lets call both and see what happens
        B::f();
        C::f();
        
        // casting this to void* then to something else entirely
        long long address = (long long)this;
        A* bad_ptr = (A*)(address + 42); // offset hacking
        
        // bad_ptr->f(); // segfault city
    }
};

void demux() {
    D d;
    d.f();
    
    // reinterpret cast of death
    float pi = 3.14159;
    int* p = reinterpret_cast<int*>(&pi);
    
    *p = 0; // destroying pi
}
