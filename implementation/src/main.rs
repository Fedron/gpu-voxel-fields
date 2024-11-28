use std::time::Duration;

use app::{ApplicationContext, State};
use camera::{Camera, CameraController, Projection};
use glium::{
    glutin::surface::WindowSurface, program::ComputeShader, uniform, Display, Surface, Texture2d,
};
use quad::ScreenQuad;
use ray::Ray;
use winit::{
    event::{DeviceEvent, KeyEvent, MouseButton, WindowEvent},
    keyboard::PhysicalKey,
    window::Window,
};
use world::{Voxel, World};

mod app;
mod camera;
mod quad;
mod ray;
mod world;

struct VoxelApp {
    camera: Camera,
    camera_controller: CameraController,
    projection: Projection,

    screen_quad: ScreenQuad,
    ray_marcher_texture: Texture2d,
    ray_marcher: ComputeShader,

    world: World,
    df_shader: ComputeShader,
}

impl ApplicationContext for VoxelApp {
    const WINDOW_TITLE: &'static str = "Voxels";

    fn new(display: &Display<WindowSurface>) -> Self {
        let camera = Camera::new(glam::Vec3::ZERO, 0.0, 0.0);
        let camera_controller = CameraController::new(20.0, 0.5);

        let (width, height) = display.get_framebuffer_dimensions();

        let projection = Projection::new(width as f32 / height as f32, 45.0, 0.1, 1000.0);

        let ray_marcher_texture = Texture2d::empty_with_format(
            display,
            glium::texture::UncompressedFloatFormat::U8U8U8U8,
            glium::texture::MipmapsOption::NoMipmap,
            width,
            height,
        )
        .expect("to create texture for ray marcher output");

        let ray_marcher =
            ComputeShader::from_source(display, include_str!("shaders/ray_marcher.comp"))
                .expect("to create ray marcher compute shader");

        let mut world = World::new(display, glam::UVec3::splat(32));
        for x in 0..world.size().x {
            for z in 0..world.size().z {
                world.set(glam::uvec3(x, 0, z), Voxel::Stone);
            }
        }

        let df_shader =
            ComputeShader::from_source(display, include_str!("shaders/distance_field.comp"))
                .expect("to create distance field compute shader");

        VoxelApp::recalculate_distance_field(&mut world, &df_shader);

        let screen_quad = ScreenQuad::new(display);

        Self {
            camera,
            camera_controller,
            projection,

            screen_quad,
            ray_marcher_texture,
            ray_marcher,

            world,
            df_shader,
        }
    }

    fn handle_window_event(
        &mut self,
        display: &Display<WindowSurface>,
        _window: &Window,
        event: &WindowEvent,
    ) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                self.camera_controller.process_keyboard(*key, *state);
            }
            WindowEvent::Resized(window_size) => {
                self.projection
                    .resize(window_size.width as f32, window_size.height as f32);

                self.ray_marcher_texture = Texture2d::empty_with_format(
                    display,
                    glium::texture::UncompressedFloatFormat::U8U8U8U8,
                    glium::texture::MipmapsOption::NoMipmap,
                    window_size.width,
                    window_size.height,
                )
                .expect("to create texture for ray marcher output");
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left && state.is_pressed() {
                    let hit = self
                        .world
                        .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
                    if hit.does_intersect {
                        self.world.set(
                            hit.voxel_position
                                .unwrap()
                                .saturating_add_signed(hit.face_normal.unwrap()),
                            Voxel::Stone,
                        );
                    }
                } else if *button == MouseButton::Right && state.is_pressed() {
                    let hit = self
                        .world
                        .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
                    if hit.does_intersect {
                        self.world.set(hit.voxel_position.unwrap(), Voxel::Air);
                    }
                }
            }
            _ => {}
        }
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.camera_controller
                    .process_mouse(delta.0 as f32, delta.1 as f32);
            }
            _ => {}
        }
    }

    fn update(&mut self, delta_time: Duration) {
        self.camera_controller
            .update_camera(&mut self.camera, delta_time.as_secs_f32());

        if self.world.is_dirty {
            VoxelApp::recalculate_distance_field(&mut self.world, &self.df_shader);
        }
    }

    fn draw_frame(&mut self, display: &glium::Display<glium::glutin::surface::WindowSurface>) {
        let mut frame = display.draw();
        frame.clear_color_srgb_and_depth((1.0, 0.0, 1.0, 1.0), 1.0);

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
            .draw(&mut frame, self.ray_marcher_texture.sampled())
            .expect("to draw ray marcher output to screen quad");

        frame.finish().expect("to finish drawing frame")
    }
}

impl VoxelApp {
    fn recalculate_distance_field(world: &mut World, shader: &ComputeShader) {
        use std::time::Instant;
        let now = Instant::now();

        shader.execute(
            uniform! {
                voxels: world.voxels_texture(),
                distance_field: world.distance_field_image().expect("to create image unit for distance field texture").set_access(glium::uniforms::ImageUnitAccess::Write),
                world_size: world.size().to_array()
            },
            world.size().x,
            world.size().y,
            world.size().z,
        );
        world.is_dirty = false;

        let elapsed = now.elapsed();
        println!("Regenerated distance field in {:.2?}", elapsed);
    }
}

fn main() {
    State::<VoxelApp>::run_loop();
}
