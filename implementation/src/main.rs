use std::rc::Rc;

use app::{App, AppBehaviour, Window};
use glium::{program::ComputeShader, uniform, Surface, Texture2d};
use winit::event::{Event, WindowEvent};

mod app;

const INITIAL_WINDOW_WIDTH: u32 = 1280;
const INITIAL_WINDOW_HEIGHT: u32 = 720;

struct VoxelApp {
    window: Rc<Window>,

    raymarched_texture: Texture2d,
    raymarcher: ComputeShader,
}

impl VoxelApp {
    fn new(window: Rc<Window>) -> Self {
        let display = &window.display;
        let raymarched_texture =
            Texture2d::empty(display, INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
                .expect("to create texture");
        let raymarcher =
            ComputeShader::from_source(display, include_str!("shaders/raymarcher.comp"))
                .expect("to create compute shader");

        Self {
            window,

            raymarched_texture,
            raymarcher,
        }
    }
}

impl AppBehaviour for VoxelApp {
    fn process_events(&mut self, event: winit::event::Event<()>) -> bool {
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                self.raymarched_texture =
                    Texture2d::empty(&self.window.display, new_size.width, new_size.height)
                        .expect("to create texture");
            }
            _ => {}
        }

        true
    }

    fn update(&mut self, _delta_time: std::time::Duration) {}

    fn render(&mut self, frame: &mut glium::Frame) {
        let out_image = self
            .raymarched_texture
            .image_unit(glium::uniforms::ImageUnitFormat::RGBA8)
            .expect("to create image unit")
            .set_access(glium::uniforms::ImageUnitAccess::Write);
        self.raymarcher.execute(
            uniform! {
                out_image: out_image,
            },
            1280,
            720,
            1,
        );

        self.raymarched_texture
            .as_surface()
            .fill(frame, glium::uniforms::MagnifySamplerFilter::Nearest);
    }
}

fn main() {
    let mut app = App::new(
        "Dynamic scene with an SVDAG",
        INITIAL_WINDOW_WIDTH,
        INITIAL_WINDOW_HEIGHT,
    );

    let voxel_app = VoxelApp::new(app.window.clone());
    app.run(voxel_app);
}
