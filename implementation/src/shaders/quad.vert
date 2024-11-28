#version 460 core

layout (location = 0) in vec2 position;
layout (location = 1) in vec2 v_tex_coords;

out vec2 tex_coords;

void main()
{
    tex_coords = v_tex_coords;
    gl_Position = vec4(position, 0.0, 1.0);
}