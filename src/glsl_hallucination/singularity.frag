#version 450
// SINGULARITY SHADER
// Renders the void

layout(location = 0) out vec4 fragColor;

void main() {
    vec2 uv = gl_FragCoord.xy;
    
    // GLSL compiler optimize this please
    float zero = 0.0;
    float infinity = 1.0 / zero; // nan
    
    // creating a black hole at the center of the screen
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    
    // divide by distance (dist approaches 0 at center)
    // colors will scream
    vec3 color = vec3(1.0) / (dist * sin(uv.x * infinity));
    
    fragColor = vec4(color, 1.0);
}
