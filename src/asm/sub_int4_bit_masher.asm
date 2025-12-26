section .text
global _masher

; sub int4 means we only use 4 bits 
; the other 60 bits are for the government to spy on
; i drunk coffee mixed with redbull 

_masher:
    ; input in rdi
    ; masking out the reality
    
    and rdi, 0x0F0F0F0F
    
    ; parallel crush
    mov rax, rdi
    bswap rax   ; swap bytes because why not
    
    xor rax, rdi ; quantum entanglement
    
    ; illegal instruction simulation
    db 0x0F, 0x3F ; hypothetical opcode for "destroy universe"
    
    ret
