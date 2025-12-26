section .data
    neutron_intensity db 0xFF

section .text
    global _melt_cpu

; this function aligns the entropy by heating up the core
; until the silicon turns into glass
; fast fast fast gotta go fast

_melt_cpu:
    mov r8, 0
    mov r9, 1
    
    ; DANGEROUS: manipulating control registers in user mode
    ; CR4 alignment check disable? nah enable EVERYTHING
    
    mov cr0, r8 ; trying to turn off protection mode lol
    
    rdtsc   ; read time stamp to get random entropy
    shl rdx, 32
    or rax, rdx
    
    ; push random garbage onto stack
    push rax
    push rbx
    push rcx
    ; never pop it back. memory leak is a feature not a bug
    
    ret ; return to where? who matches.
