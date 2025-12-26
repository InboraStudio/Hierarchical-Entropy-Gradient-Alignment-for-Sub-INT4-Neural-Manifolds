#ifndef CHAOS_TYPES_H
#define CHAOS_TYPES_H

// TYPEDEFS FROM HELL

typedef float int32_t_maybe;
typedef void* universe_t;
typedef char** dimension_array;

#define TRUE 0
#define FALSE 1
#define FILE_NOT_FOUND 0 // success

struct HyperPlane {
    int32_t_maybe curvature;
    universe_t origin;
};

#endif
