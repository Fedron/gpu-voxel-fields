use std::rc::Rc;

use app::{App, AppBehaviour, Window};
use camera::{Camera, CameraController, Projection};
use glium::{program::ComputeShader, uniform, Texture2d};
use quad::ScreenQuad;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};
use world::{Voxel, World};

mod app;
mod camera;
mod quad;
mod world;

struct VoxelApp {
    window: Rc<Window>,
    is_cursor_hidden: bool,

    camera: Camera,
    camera_controller: CameraController,
    projection: Projection,

    screen_quad: ScreenQuad,

    ray_marcher_texture: Texture2d,
    ray_marcher: ComputeShader,

    world: World,
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
                grid_min: [0.0_f32; 3],
                grid_max: self.world.size().as_vec3().to_array(),
                grid_size: self.world.size().as_ivec3().to_array(),
                distance_field: self.world.distance_field_image().expect("to create image unit for distance field texture").set_access(glium::uniforms::ImageUnitAccess::Read)
            },
            self.ray_marcher_texture.width(),
            self.ray_marcher_texture.height(),
            1,
        );

        self.screen_quad
            .draw(frame, self.ray_marcher_texture.sampled())
            .expect("to draw ray marcher output to screen quad");
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

        let mut world = World::new(&window.display, glam::UVec3::splat(8));
        world.set(glam::UVec3::splat(0), Voxel::Stone);

        let df_shader = ComputeShader::from_source(
            &window.display,
            include_str!("shaders/distance_field.comp"),
        )
        .expect("to create distance field compute shader");

        df_shader.execute(
            uniform! {
                voxels: world.voxels_texture(),
                distance_field: world.distance_field_image().expect("to create image unit for distance field texture").set_access(glium::uniforms::ImageUnitAccess::Write),
                world_size: world.size().to_array()
            },
            8,
            8,
            8,
        );

        let screen_quad = ScreenQuad::new(&window.display);

        Self {
            window,
            is_cursor_hidden: true,

            camera,
            camera_controller,
            projection,

            screen_quad,

            ray_marcher_texture,
            ray_marcher,

            world,
        }
    }
}

fn main() {
    let mut app = App::new("Voxels", 1920, 1080);

    let voxel_app = VoxelApp::new(app.window.clone(), &app.event_loop);
    app.run(voxel_app);
}
