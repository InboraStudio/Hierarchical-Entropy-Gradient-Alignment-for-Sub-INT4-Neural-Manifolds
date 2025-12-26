use std::mem;
use std::ptr;
use std::slice;
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

// UNSAFE RUST - BORROW CHECKER EVASION TECHNIQUES
// this is what happens when you tell rust "no"
// all safety guarantees voided

static mut GLOBAL_CHAOS_BUFFER: [u8; 1024 * 1024] = [0; 1024 * 1024];
static ALLOCATION_COUNTER: AtomicUsize = AtomicUsize::new(0);

// struct with self-referential pointer (normally impossible)
struct SelfReference {
    data: Vec<u8>,
    ptr_to_self: *const Vec<u8>,
}

impl SelfReference {
    unsafe fn new() -> Self {
        let mut s = SelfReference {
            data: vec![0u8; 1024],
            ptr_to_self: ptr::null(),
        };
        // create self reference (dangling pointer waiting to happen)
        s.ptr_to_self = &s.data as *const Vec<u8>;
        s
    }
    
    unsafe fn corrupt(&mut self) {
        // dereference potentially dangling pointer
        let data_ref = &*self.ptr_to_self;
        println!("Data length: {}", data_ref.len());
    }
}

// transmute lifetime from local to static
unsafe fn transmute_lifetime<'a, T>(r: &'a T) -> &'static T {
    mem::transmute::<&'a T, &'static T>(r)
}

// create dangling pointer
unsafe fn create_dangling() -> &'static i32 {
    let x = 42;
    transmute_lifetime(&x)  // x dies, reference lives
}

// manual memory management
struct ManualAlloc {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
}

impl ManualAlloc {
    unsafe fn new(size: usize) -> Self {
        let layout = Layout::from_size_align_unchecked(size, 64);
        let ptr = alloc(layout);
        
        // write pattern to allocated memory
        for i in 0..size {
            *ptr.add(i) = (i & 0xFF) as u8;
        }
        
        ManualAlloc { ptr, size, layout }
    }
    
    unsafe fn write_unchecked(&mut self, offset: usize, value: u8) {
        // no bounds checking
        *self.ptr.add(offset) = value;
    }
    
    unsafe fn read_unchecked(&self, offset: usize) -> u8 {
        *self.ptr.add(offset)
    }
    
    unsafe fn double_free(&mut self) {
        dealloc(self.ptr, self.layout);
        dealloc(self.ptr, self.layout);  // double free
    }
}

impl Drop for ManualAlloc {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

// data race generator
fn data_race_chaos() {
    use std::thread;
    
    unsafe {
        let mut threads = vec![];
        
        // spawn 10 threads all writing to global buffer
        for i in 0..10 {
            threads.push(thread::spawn(move || {
                for j in 0..100000 {
                    let idx = (i * 100000 + j) % GLOBAL_CHAOS_BUFFER.len();
                    GLOBAL_CHAOS_BUFFER[idx] = (i ^ j) as u8;
                }
            }));
        }
        
        // join threads
        for t in threads {
            t.join().unwrap();
        }
    }
}

// type confusion via transmute
unsafe fn type_confusion() {
    let x: f64 = 3.14159;
    
    // reinterpret float as integer
    let x_bits: u64 = mem::transmute(x);
    println!("Float as bits: 0x{:016X}", x_bits);
    
    // transmute to pointer
    let weird_ptr: *const u8 = mem::transmute(x_bits);
    println!("Float as pointer: {:p}", weird_ptr);
    
    // try to dereference (crash)
    // let byte = *weird_ptr;
}

// union for type punning
union TypePun {
    i: i64,
    f: f64,
    p: *mut u8,
    bytes: [u8; 8],
}

unsafe fn type_pun_chaos() {
    let mut pun = TypePun { i: 0xDEADBEEFCAFEBABE };
    
    // read as float
    let f = pun.f;
    println!("Int as float: {}", f);
    
    // read as pointer
    let p = pun.p;
    println!("Int as pointer: {:p}", p);
    
    // write through pointer (crash likely)
    // *p = 0x42;
}

// buffer overflow
unsafe fn buffer_overflow() {
    let mut buffer = vec![0u8; 16];
    let ptr = buffer.as_mut_ptr();
    
    // write beyond buffer bounds
    for i in 0..1024 {
        *ptr.add(i) = (i & 0xFF) as u8;
    }
}

// use after free
unsafe fn use_after_free() {
    let layout = Layout::from_size_align_unchecked(1024, 8);
    let ptr = alloc(layout);
    
    // write to allocated memory
    for i in 0..1024 {
        *ptr.add(i) = 0xAA;
    }
    
    // free it
    dealloc(ptr, layout);
    
    // use after free
    for i in 0..1024 {
        let val = *ptr.add(i);
        *ptr.add(i) = val ^ 0xFF;
    }
}

// null pointer dereference
unsafe fn null_deref() {
    let null: *mut i32 = ptr::null_mut();
    // *null = 42;  // uncomment to crash
}

// uninitialized memory
unsafe fn uninit_memory() {
    let x: i32 = mem::uninitialized();  // deprecated but still works
    println!("Uninitialized value: {}", x);  // undefined behavior
}

// raw pointer arithmetic
unsafe fn pointer_arithmetic_chaos() {
    let arr = vec![1, 2, 3, 4, 5];
    let ptr = arr.as_ptr();
    
    // walk off the end of the array
    for i in 0..1000 {
        let val = *ptr.add(i);
        println!("arr[{}] = {}", i, val);
    }
}

// mutable aliasing
unsafe fn mutable_aliasing() {
    let mut x = 42;
    let r1 = &mut x as *mut i32;
    let r2 = &mut x as *mut i32;
    
    // two mutable aliases (undefined behavior)
    *r1 = 100;
    *r2 = 200;
    
    println!("x = {}", x);
}

// infinite recursion with unsafe
unsafe fn infinite_recursion(depth: usize) {
    if depth > 1000000 {
        println!("Stack overflow imminent");
    }
    
    // allocate on each recursion to eat stack faster
    let _local = [0u8; 1024];
    
    infinite_recursion(depth + 1);
}

// ffi abuse
extern "C" {
    fn undefined_function() -> i32;  // doesn't exist
}

unsafe fn call_undefined_function() {
    // let result = undefined_function();  // will crash
}

// violate lifetime constraints
struct Container<'a> {
    reference: &'a i32,
}

unsafe fn lifetime_violation() -> Container<'static> {
    let x = 42;
    Container {
        reference: mem::transmute(&x),
    }
}

// main chaos coordinator
fn main() {
    println!("[UNSAFE RUST] Initializing chaos engine...");
    
    unsafe {
        // global buffer corruption
        for i in 0..GLOBAL_CHAOS_BUFFER.len() {
            GLOBAL_CHAOS_BUFFER[i] = (i ^ 0xAA) as u8;
        }
        println!("[UNSAFE RUST] Global buffer corrupted");
        
        // data race
        println!("[UNSAFE RUST] Starting data race...");
        data_race_chaos();
        
        // type confusion
        println!("[UNSAFE RUST] Type confusion...");
        type_confusion();
        type_pun_chaos();
        
        // manual allocation
        println!("[UNSAFE RUST] Manual allocation...");
        let mut manual = ManualAlloc::new(4096);
        for i in 0..4096 {
            manual.write_unchecked(i, 0xBB);
        }
        
        // buffer overflow
        println!("[UNSAFE RUST] Buffer overflow...");
        // buffer_overflow();  // uncomment to crash
        
        // use after free
        println!("[UNSAFE RUST] Use after free...");
        // use_after_free();  // uncomment to crash
        
        // dangling pointer
        println!("[UNSAFE RUST] Creating dangling pointer...");
        let dangling = create_dangling();
        // println!("Dangling value: {}", *dangling);  // undefined
        
        // self reference
        println!("[UNSAFE RUST] Self-referential struct...");
        let mut self_ref = SelfReference::new();
        self_ref.corrupt();
        
        // mutable aliasing
        println!("[UNSAFE RUST] Mutable aliasing...");
        mutable_aliasing();
        
        // infinite recursion
        println!("[UNSAFE RUST] Starting infinite recursion...");
        // infinite_recursion(0);  // uncomment for stack overflow
        
        println!("[UNSAFE RUST] Chaos complete. Memory safety destroyed.");
    }
}
