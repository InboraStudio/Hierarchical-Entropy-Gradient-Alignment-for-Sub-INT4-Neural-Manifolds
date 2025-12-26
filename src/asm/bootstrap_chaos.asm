section .data
    align 64
    magic_constants: dq 0xDEADBEEFDEAD BEEF, 0xCAFEBABECAFEBABE
    chaos_seed: dq 0x0BADF00D0BADF00D
    entropy_pool: times 256 dq 0
    
section .bss
    align 4096
    scratch_buffer: resb 8192

section .text
    global _start
    global _chaos_entry
    global _self_modify
    global _register_havoc

; bootstrap chaos - main entry point
; this is where sanity goes to die
; typing with one hand holding coffee

_start:
    ; save all registers because we might corrupt them
    push rax
    push rbx
    push rcx
    push rdx
    push rsi
    push rdi
    push rbp
    push r8
    push r9
    push r10
    push r11
    push r12
    push r13
    push r14
    push r15
    
    ; align stack to 64 byte boundary for avx
    mov rbp, rsp
    and rsp, -64
    
    ; initialize entropy pool with rdtsc
    xor rcx, rcx
.init_entropy:
    rdtsc
    shl rdx, 32
    or rax, rdx
    mov [entropy_pool + rcx * 8], rax
    inc rcx
    cmp rcx, 256
    jl .init_entropy
    
    ; call the chaos engine
    call _chaos_entry
    
    ; restore stack
    mov rsp, rbp
    
    ; restore registers in wrong order for chaos
    pop r15
    pop r14
    pop r13
    pop r12
    pop r11
    pop r10
    pop r9
    pop r8
    pop rbp
    pop rdi
    pop rsi
    pop rdx
    pop rcx
    pop rbx
    pop rax
    
    ; exit with cursed code
    mov rax, 60
    mov rdi, 0xBADC0DE
    syscall

; chaos entry - the meat grinder
_chaos_entry:
    ; load magic constants
    mov rax, [magic_constants]
    mov rbx, [magic_constants + 8]
    mov rcx, [chaos_seed]
    
    ; initialize loop counter to max
    mov r15, 0xFFFFFFFFFFFFFFFF
    
.main_loop:
    ; rotate and mix
    rol rax, 7
    ror rbx, 13
    xor rcx, rax
    xor rcx, rbx
    
    ; add with carry chain
    add rax, rcx
    adc rbx, rax
    adc rcx, rbx
    
    ; multiply by prime
    mov rdx, 0x9E3779B97F4A7C15
    imul rax, rdx
    
    ; conditional chaos
    test rax, 0xFF
    jz .trigger_chaos
    
.continue_loop:
    ; decrement and loop
    dec r15
    jnz .main_loop
    jmp .cleanup
    
.trigger_chaos:
    ; save state
    push rax
    push rbx
    push rcx
    
    ; call self modification routine
    call _self_modify
    
    ; call register havoc
    call _register_havoc
    
    ; restore state
    pop rcx
    pop rbx
    pop rax
    jmp .continue_loop

.cleanup:
    ; write final state to scratch buffer
    mov rdi, scratch_buffer
    mov [rdi], rax
    mov [rdi + 8], rbx
    mov [rdi + 16], rcx
    
    ret

; self modifying code section
; rewrites its own instructions
_self_modify:
    ; get address of target
    lea rsi, [rel .target]
    
    ; write nop sled
    mov byte [rsi], 0x90
    mov byte [rsi + 1], 0x90
    mov byte [rsi + 2], 0x90
    mov byte [rsi + 3], 0x90
    mov byte [rsi + 4], 0x90
    
    ; write a ret after nops
    mov byte [rsi + 5], 0xC3
    
    ; flush instruction cache (x86 is coherent but lets be paranoid)
    mfence
    sfence
    
.target:
    ; this will be overwritten
    xor rax, rax
    xor rbx, rbx
    xor rcx, rcx
    
    ret

; register destruction sequence
_register_havoc:
    ; corrupt all general purpose registers
    xor rax, rax
    dec rax              ; rax = 0xFFFFFFFFFFFFFFFF
    
    mov rbx, rax
    shr rbx, 1           ; rbx = 0x7FFFFFFFFFFFFFFF
    
    mov rcx, rax
    rol rcx, 13
    
    mov rdx, 0xDEADBEEFCAFEBABE
    
    mov rsi, 0xBADF00D0BADF00D
    mov rdi, rsi
    not rdi
    
    ; r8-r15 get progressively more cursed values
    mov r8, 0x0123456789ABCDEF
    bswap r8             ; byte swap for chaos
    
    mov r9, r8
    imul r9, r9          ; square it
    
    mov r10, ~0
    mov r11, 0
    
    ; use rdrand if available (will fail on old cpus)
    rdrand r12
    rdrand r13
    rdrand r14
    rdrand r15
    
    ; mix everything together
    xor rax, r8
    xor rbx, r9
    xor rcx, r10
    xor rdx, r11
    xor rsi, r12
    xor rdi, r13
    
    ret

; section dedicated to port io chaos
; writes to weird ports
section .text
_port_chaos:
    ; write to port 0x80 (POST code)
    mov al, 0xAA
    out 0x80, al
    
    ; write to port 0x70 (CMOS)
    mov al, 0x0D
    out 0x70, al
    
    ; read from port 0x71
    in al, 0x71
    
    ; write to port 0x3F8 (serial)
    mov dx, 0x3F8
    mov al, 0x42
    out dx, al
    
    ret

; cpu feature detection and abuse
_cpuid_chaos:
    ; get cpu features
    mov eax, 1
    cpuid
    
    ; check for avx
    and ecx, (1 << 28)
    jz .no_avx
    
    ; corrupt ymm registers if avx available
    vxorps ymm0, ymm0, ymm0
    vpcmpeqq ymm1, ymm1, ymm1    ; set all bits
    
.no_avx:
    ; check for aes-ni
    mov eax, 1
    cpuid
    and ecx, (1 << 25)
    jz .no_aes
    
    ; use aes instruction for chaos
    aesenc xmm0, xmm1
    
.no_aes:
    ret

; massive unrolled loop
; this will bloat the binary
_unrolled_chaos:
    %assign i 0
    %rep 100
        mov rax, i
        rol rax, i % 64
        xor rbx, rax
        add rcx, rbx
        %assign i i+1
    %endrep
    
    ret

; stack corruption routine
_stack_chaos:
    ; save original rsp
    mov rbx, rsp
    
    ; allocate massive stack frame
    sub rsp, 0x10000
    
    ; fill stack with pattern
    mov rdi, rsp
    mov rcx, 0x2000
    mov rax, 0xDEADC0DEDEADC0DE
    rep stosq
    
    ; corrupt stack pointer
   mov rsp, rbx
    add rsp, 0x42        ; misalign it
    
    ; restore (maybe)
    mov rsp, rbx
    
    ret

; exception generator
; triggers various cpu exceptions
_exception_chaos:
    ; divide by zero
    xor rdx, rdx
    xor rax, rax
    mov rcx, 0
    ; div rcx              ; uncomment to crash
    
    ; invalid opcode
    ; db 0x0F, 0x0B       ; ud2 instruction
    
    ; general protection fault
    ; mov cr0, rax        ; requires ring 0
    
    ret

; final chaos - infinite loop of madness
_infinite_chaos:
    mov rcx, 1000000
.loop:
    call _register_havoc
    call _self_modify
    call _cpuid_chaos
    call _unrolled_chaos
    call _stack_chaos
    
    dec rcx
    jnz .loop
    
    ; halt and catch fire
    hlt
    jmp _infinite_chaos
