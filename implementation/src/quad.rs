use glium::{
    glutin::surface::WindowSurface, implement_vertex, uniform, uniforms::Sampler, Display,
    DrawError, Program, Surface, Texture2d, VertexBuffer,
};

#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    v_tex_coords: [f32; 2],
}
implement_vertex!(Vertex, position, v_tex_coords);

pub struct ScreenQuad {
    program: Program,
    vertices: VertexBuffer<Vertex>,
}

impl ScreenQuad {
    pub fn new(display: &Display<WindowSurface>) -> Self {
        let program = Program::from_source(
            display,
            include_str!("shaders/quad.vert"),
            include_str!("shaders/quad.frag"),
            None,
        )
        .expect("to create quad program");

        let vertices = vec![
            Vertex {
                position: [-1.0, 1.0],
                v_tex_coords: [0.0, 1.0],
            },
            Vertex {
                position: [-1.0, -1.0],
                v_tex_coords: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0],
                v_tex_coords: [1.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0],
                v_tex_coords: [1.0, 0.0],
            },
        ];

        let vertices =
            VertexBuffer::new(display, &vertices).expect("to create vertex buffer for screen quad");

        Self { program, vertices }
    }

    pub fn draw(
        &self,
        frame: &mut glium::Frame,
        sampler: Sampler<Texture2d>,
    ) -> Result<(), DrawError> {
        frame.draw(
            &self.vertices,
            glium::index::NoIndices(glium::index::PrimitiveType::TriangleStrip),
            &self.program,
            &uniform! {
                tex: sampler
            },
            &Default::default(),
        )
    }
}
