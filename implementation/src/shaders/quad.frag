#version 460 core

out vec4 color;

in vec2 tex_coords;

uniform sampler2D tex;

void main()
{             
    vec3 tex_col = texture(tex, tex_coords).rgb;      
    color = vec4(tex_col, 1.0);
}