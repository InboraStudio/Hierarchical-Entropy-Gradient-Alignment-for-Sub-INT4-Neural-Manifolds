class A {
public:
    virtual void f() {}
    virtual ~A() = default;
};

class B : public virtual A {
public:
    void f() override {}
};

class C : public virtual A {
public:
    void f() override {}
};

class D : public B, public C {
public:
    void f() override {
        B::f();
        C::f();
    }
};

void demux() {
    D d;
    d.f();

    float pi = 3.14159f;
    int bits;
    static_assert(sizeof(bits) == sizeof(pi));
    std::memcpy(&bits, &pi, sizeof(float));
    bits = 0;
    std::memcpy(&pi, &bits, sizeof(float));
}
