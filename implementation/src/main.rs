use std::rc::Rc;

use app::{App, AppBehaviour, Window};
use camera::{Camera, CameraController, Projection};
use glium::{program::ComputeShader, uniform, uniforms::UniformBuffer, Surface, Texture2d};
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};
use world::{DistanceField, VoxelGrid};

mod app;
mod camera;
mod world;

struct VoxelApp {
    window: Rc<Window>,
    is_cursor_hidden: bool,

    camera: Camera,
    camera_controller: CameraController,
    projection: Projection,

    ray_marcher_texture: Texture2d,
    ray_marcher: ComputeShader,

    distance_field: UniformBuffer<DistanceField>,
}

impl AppBehaviour for VoxelApp {
    fn process_events(&mut self, event: winit::event::Event<()>) -> bool {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            state: ElementState::Pressed,
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            ..
                        },
                    ..
                } => false,
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(key),
                            state,
                            ..
                        },
                    ..
                } => {
                    if key == KeyCode::AltLeft && state == ElementState::Pressed {
                        self.is_cursor_hidden = false;
                    } else if key == KeyCode::AltLeft && state == ElementState::Released {
                        self.is_cursor_hidden = true;
                    }

                    self.camera_controller.process_keyboard(key, state);
                    true
                }
                WindowEvent::Resized(window_size) => {
                    self.projection
                        .resize(window_size.width as f32, window_size.height as f32);

                    self.ray_marcher_texture = Texture2d::empty_with_format(
                        &self.window.display,
                        glium::texture::UncompressedFloatFormat::U8U8U8U8,
                        glium::texture::MipmapsOption::NoMipmap,
                        window_size.width,
                        window_size.height,
                    )
                    .expect("to create texture for ray marcher output");

                    true
                }
                _ => true,
            },
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if self.is_cursor_hidden {
                    self.camera_controller
                        .process_mouse(delta.0 as f32, delta.1 as f32);
                }

                true
            }
            _ => true,
        }
    }

    fn update(&mut self, delta_time: std::time::Duration) {
        self.camera_controller
            .update_camera(&mut self.camera, delta_time.as_secs_f32());
    }

    fn render(&mut self, frame: &mut glium::Frame) {
        self.window.winit.set_cursor_visible(!self.is_cursor_hidden);

        let ray_marcher_output_image = self
            .ray_marcher_texture
            .image_unit(glium::uniforms::ImageUnitFormat::RGBA8)
            .expect("to create image unit from ray marcher texture")
            .set_access(glium::uniforms::ImageUnitAccess::Write);

        let inverse_view = self.camera.view_matrix().inverse().to_cols_array_2d();
        let inverse_projection = self.projection.matrix().inverse().to_cols_array_2d();

        self.ray_marcher.execute(
            uniform! {
                output_image: ray_marcher_output_image,
                camera_position: self.camera.position.to_array(),
                inverse_view: inverse_view,
                inverse_projection: inverse_projection,
                grid_min: [0.0_f32, 0.0_f32, 0.0_f32],
                grid_max: [8.0_f32, 8.0_f32, 8.0_f32],
                grid_size: [8_i32, 8_i32, 8_i32],
                DistanceField: &*self.distance_field
            },
            self.ray_marcher_texture.width(),
            self.ray_marcher_texture.height(),
            1,
        );

        self.ray_marcher_texture
            .as_surface()
            .fill(frame, glium::uniforms::MagnifySamplerFilter::Nearest);
    }
}

impl VoxelApp {
    fn new(window: Rc<Window>, _event_loop: &winit::event_loop::EventLoop<()>) -> Self {
        window
            .winit
            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
            .or_else(|_| {
                window
                    .winit
                    .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            })
            .expect("to lock cursor to window");
        window.winit.set_cursor_visible(false);

        let camera = Camera::new(glam::Vec3::ZERO, 0.0, 0.0);
        let camera_controller = CameraController::new(20.0, 0.5);

        let window_size = window.winit.inner_size();

        let projection = Projection::new(
            window_size.width as f32 / window_size.height as f32,
            45.0,
            0.1,
            1000.0,
        );

        let ray_marcher_texture = Texture2d::empty_with_format(
            &window.display,
            glium::texture::UncompressedFloatFormat::U8U8U8U8,
            glium::texture::MipmapsOption::NoMipmap,
            window_size.width,
            window_size.height,
        )
        .expect("to create texture for ray marcher output");

        let ray_marcher =
            ComputeShader::from_source(&window.display, include_str!("shaders/ray_marcher.comp"))
                .expect("to create ray marcher compute shader");

        let mut voxel_grid_buffer: UniformBuffer<VoxelGrid> =
            UniformBuffer::empty(&window.display).expect("to create buffer for voxel grid");

        {
            let mut mapping = voxel_grid_buffer.map();
            for x in 0..8 {
                for y in 0..8 {
                    for z in 0..8 {
                        mapping.set(
                            glam::uvec3(x, y, z),
                            if y == 0 {
                                world::Voxel::Stone
                            } else {
                                world::Voxel::Air
                            },
                        )
                    }
                }
            }
        }

        let distance_field: UniformBuffer<DistanceField> =
            UniformBuffer::empty(&window.display).expect("to create buffer for distance field");

        let df_shader = ComputeShader::from_source(
            &window.display,
            include_str!("shaders/distance_field.comp"),
        )
        .expect("to create distance field compute shader");

        let world_size: [u32; 4] = [8; 4];
        df_shader.execute(
            uniform! {
                World: &*voxel_grid_buffer,
                DistanceField: &*distance_field,
                world_size: world_size
            },
            1,
            1,
            1,
        );

        Self {
            window,
            is_cursor_hidden: true,

            camera,
            camera_controller,
            projection,

            ray_marcher_texture,
            ray_marcher,

            distance_field,
        }
    }
}

fn main() {
    let mut app = App::new("Voxels", 1920, 1080);

    let voxel_app = VoxelApp::new(app.window.clone(), &app.event_loop);
    app.run(voxel_app);
}
