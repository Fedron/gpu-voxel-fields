#![feature(generic_const_exprs)]
#![feature(duration_millis_float)]
#![feature(map_try_insert)]
#![feature(variant_count)]
#![allow(incomplete_features)]

use app::{App, AppState};
use camera::{Camera, CameraController};
use distance_field::DistanceFieldPipeline;
use egui::{vec2, Align, Align2, Color32, CornerRadius, Frame, Margin, Shadow, Window};
use egui_winit_vulkano::{Gui, GuiConfig};
use place_over_frame::RenderPassPlaceOverFrame;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use ray::Ray;
use ray_marcher_pipeline::RayMarcherPipeline;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use utils::{
    get_bool_input, get_sphere_positions, get_u64_input, get_usize_input_power_of_2, Statistics,
};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    sync::GpuFuture,
};
use vulkano_util::{context::VulkanoContext, renderer::VulkanoWindowRenderer};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};
use world::{voxel::Voxel, World};

mod app;
mod camera;
mod crosshair_pipeline;
mod distance_field;
mod pixels_draw_pipeline;
mod place_over_frame;
mod ray;
mod ray_marcher_pipeline;
mod utils;
mod world;

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::<VoxelsApp>::new(&event_loop);

    println!("Using {}\n", app.context.device_name());

    println!(
        "\
Fast GPU Generation of Signed Distance Fields from a Voxel Grid.

Voxel world rendered using a dynamically updated discrete distance field. The world can be interacted with in real-time
by placing and destroying blocks.

Usage:
    Esc: Quit

    LMB: Place active voxel
    RMB: Delete voxel
    Scroll (+/-): Increase/decrease brush size

    1: Set 'Stone' as active voxel
    2: Set 'Sand' as active voxel
    3: Set 'Water as active voxel
        \n",
    );

    event_loop.run_app(&mut app)?;

    let avg_dt =
        app.frame_stats.frame_times.iter().sum::<f64>() / app.frame_stats.frame_times.len() as f64;
    println!(
        "\nAverage FPS: {:.5}\nAverage Delta Time (ms): {:.5}\n",
        1000.0 / avg_dt,
        avg_dt
    );

    let state = app.state.unwrap();
    let avg_dt = state.ddf_generation_stats.frame_times.iter().sum::<f64>()
        / state.ddf_generation_stats.frame_times.len() as f64;
    println!("Average FPS during regeneration: {:.5}", 1000.0 / avg_dt);
    println!("Average Delta Time during regeneration: {:.5}\n", avg_dt);

    let stats = Statistics::calculate(&state.ddf_generation_stats.df_execution_times);
    println!(
        "DDF Algorithm Runtime Statistics (ms) ({} entries)\n{:#?}\n",
        &state.ddf_generation_stats.df_execution_times.len(),
        stats
    );

    println!(
        "Average DDF regenerations per second: {:.5}/s\n",
        state.ddf_generation_stats.df_execution_times.len() as f64
            / app.frame_stats.start_time.elapsed().as_secs_f64()
    );

    println!(
        "DDF Regeneration Events: {}\nWorld Updates: {}\n",
        state.ddf_generation_stats.df_execution_times.len(),
        state.world.update_count
    );

    let stats = Statistics::calculate(&state.ddf_generation_stats.rm_execution_times);
    println!(
        "Ray Marcher Runtime Statistics (ms) ({} entries)\n{:#?}\n",
        &state.ddf_generation_stats.rm_execution_times.len(),
        stats
    );

    Ok(())
}

#[derive(Clone, Copy)]
struct VoxelAppConfiguration {
    test_mode: bool,
    seed: u64,
    modification_interval: Duration,
    world_size: usize,
    chunk_size: usize,
    focal_size: u64,
}

impl VoxelAppConfiguration {
    fn from_input() -> Self {
        let test_mode = get_bool_input("Run the application in test mode?", true);
        let seed = get_u64_input("Enter RNG Seed (u64)", 6683787);
        let modification_interval =
            get_u64_input("Enter world modification interval (milliseconds)", 200);
        let world_size = get_usize_input_power_of_2("Enter world size (usize)", 128, None);
        let chunk_size = get_usize_input_power_of_2(
            "Enter chunk size (must be smaller than world size)",
            16,
            Some(world_size),
        );
        let focal_size = get_u64_input("Enter a focal size (u64)", 1);

        Self {
            test_mode,
            seed,
            modification_interval: Duration::from_millis(modification_interval),
            world_size,
            chunk_size,
            focal_size,
        }
    }
}

struct VoxelsApp {
    configuration: VoxelAppConfiguration,

    distance_field_pipeline: DistanceFieldPipeline,
    ray_marcher_pipeline: RayMarcherPipeline,
    place_over_frame: RenderPassPlaceOverFrame,
    gui: Gui,

    camera: Camera,
    camera_controller: CameraController,
    world: World,
    distance_fields: Vec<Subbuffer<[u8]>>,

    ddf_generation_stats: DDFGenerationStats,
    push_delta_time: bool,

    rng: SmallRng,
    last_modification: Instant,

    num_chunks: glam::UVec3,
    voxel_to_place: Voxel,
    lmb_held: bool,
    rmb_held: bool,
    brush_size: u32,
    alt_held: bool,
}

impl AppState for VoxelsApp {
    const WINDOW_TITLE: &'static str = "Voxels";

    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        context: &VulkanoContext,
        window_renderer: &VulkanoWindowRenderer,
    ) -> Self {
        let configuration = VoxelAppConfiguration::from_input();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        let num_chunks = glam::UVec3::splat(configuration.world_size as u32)
            .saturating_div(glam::UVec3::splat(configuration.chunk_size as u32));
        let mut world = World::new(
            glam::UVec3::splat(configuration.chunk_size as u32),
            num_chunks,
            context.memory_allocator().clone(),
        );

        for x in 0..world.size().x {
            for z in 0..world.size().z {
                world.set(glam::uvec3(x, 0, z), Voxel::Stone);
            }
        }
        world.update_count = 1;

        let distance_field_pipeline = DistanceFieldPipeline::new(
            context.compute_queue().clone(),
            context.memory_allocator().clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            configuration.chunk_size as u32,
            (num_chunks / 2).saturating_sub(glam::UVec3::ONE),
            configuration.focal_size as u32,
        );

        let mut distance_fields = Vec::with_capacity(num_chunks.element_product() as usize);
        for _ in 0..num_chunks.element_product() {
            distance_fields.push(distance_field_pipeline.allocate_distance_field());
        }

        let ray_marcher_pipeline = RayMarcherPipeline::new(
            context.graphics_queue().clone(),
            context.memory_allocator().clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            distance_fields.clone(),
            &world,
        );

        let place_over_frame = RenderPassPlaceOverFrame::new(
            context.graphics_queue().clone(),
            context.memory_allocator().clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            window_renderer.swapchain_format(),
            window_renderer.swapchain_image_views(),
        );

        let gui = Gui::new_with_subpass(
            event_loop,
            window_renderer.surface(),
            window_renderer.graphics_queue(),
            place_over_frame.gui_subpass(),
            window_renderer.swapchain_format(),
            GuiConfig::default(),
        );

        Self {
            configuration,

            distance_field_pipeline,
            ray_marcher_pipeline,
            place_over_frame,
            gui,

            camera: Camera::new(
                glam::vec3(
                    -(world.size().x as f32),
                    world.size().y as f32 / 2.0,
                    world.size().z as f32 / 2.0,
                ),
                0.0,
                0.0,
                60.0_f32.to_radians(),
                (
                    window_renderer.window_size()[0] as u32,
                    window_renderer.window_size()[1] as u32,
                ),
            ),
            camera_controller: CameraController::new(10.0, 1.0),
            world,
            distance_fields,

            ddf_generation_stats: DDFGenerationStats {
                df_execution_times: Vec::new(),
                rm_execution_times: Vec::new(),
                frame_times: Vec::new(),
            },
            push_delta_time: false,

            rng: SmallRng::seed_from_u64(configuration.seed),
            last_modification: Instant::now(),

            num_chunks,
            voxel_to_place: Voxel::Stone,
            lmb_held: false,
            rmb_held: false,
            brush_size: 1,
            alt_held: false,
        }
    }

    fn handle_window_event(
        &mut self,
        window: &winit::window::Window,
        event_loop: &ActiveEventLoop,
        event: &WindowEvent,
    ) {
        if self.gui.update(event) {
            return;
        }

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

                match *key {
                    KeyCode::Escape => {
                        if state.is_pressed() {
                            event_loop.exit();
                        }
                    }
                    KeyCode::Digit1 => self.voxel_to_place = Voxel::Stone,
                    KeyCode::Digit2 => self.voxel_to_place = Voxel::Sand,
                    KeyCode::Digit3 => self.voxel_to_place = Voxel::Water,
                    KeyCode::AltLeft => self.alt_held = state.is_pressed(),
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match (*button, *state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    if !self.lmb_held {
                        self.lmb_held = true;
                    }
                }
                (MouseButton::Left, ElementState::Released) => {
                    if self.lmb_held {
                        self.lmb_held = false;
                    }
                }
                (MouseButton::Right, ElementState::Pressed) => {
                    if !self.rmb_held {
                        self.rmb_held = true;
                    }
                }
                (MouseButton::Right, ElementState::Released) => {
                    if self.rmb_held {
                        self.rmb_held = false;
                    }
                }
                _ => {}
            },
            WindowEvent::Resized(new_size) => self.camera.resize((new_size.width, new_size.height)),
            _ => {}
        }

        window.set_cursor_visible(self.alt_held);
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => self
                .camera_controller
                .process_mouse(delta.0 as f32, delta.1 as f32),
            DeviceEvent::MouseWheel { delta } => {
                if let MouseScrollDelta::LineDelta(_, y) = delta {
                    self.brush_size = self.brush_size.saturating_add_signed(*y as i32);
                }
            }
            _ => {}
        }
    }

    fn update(&mut self, delta_time: Duration) {
        if self.push_delta_time {
            self.push_delta_time = false;
            self.ddf_generation_stats
                .frame_times
                .push(delta_time.as_secs_f64() * 1000.0)
        }

        if self.configuration.test_mode
            && self.last_modification.elapsed() > self.configuration.modification_interval
        {
            let positions = get_sphere_positions(
                glam::ivec3(
                    self.rng.gen_range(0..self.world.size().x) as i32,
                    self.rng.gen_range(0..self.world.size().y) as i32,
                    self.rng.gen_range(0..self.world.size().z) as i32,
                ),
                self.rng
                    .gen_range(1..(self.configuration.chunk_size as u32)),
            );
            for &position in positions.iter() {
                if self.world.is_in_bounds_ivec3(position) {
                    self.world.set(
                        position.as_uvec3(),
                        (self.rng.gen_range(0..std::mem::variant_count::<Voxel>()) as u32).into(),
                    );
                }
            }
        }

        if !self.alt_held {
            self.camera_controller
                .update_camera(&mut self.camera, delta_time.as_secs_f32());
        }

        if self.lmb_held {
            let hit = self
                .world
                .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
            if hit.does_intersect {
                let hit_position = hit
                    .voxel_position
                    .unwrap()
                    .saturating_add_signed(hit.face_normal.unwrap());

                let positions = get_sphere_positions(hit_position.as_ivec3(), self.brush_size);
                for &position in positions.iter() {
                    if self.world.is_in_bounds_ivec3(position) {
                        self.world.set(position.as_uvec3(), self.voxel_to_place);
                    }
                }
            }
        } else if self.rmb_held {
            let hit = self
                .world
                .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
            if hit.does_intersect {
                let positions =
                    get_sphere_positions(hit.voxel_position.unwrap().as_ivec3(), self.brush_size);
                for &position in positions.iter() {
                    if self.world.is_in_bounds_ivec3(position) {
                        self.world.set(position.as_uvec3(), Voxel::Air);
                    }
                }
            }
        }

        for (index, chunk) in self
            .world
            .chunks
            .iter_mut()
            .enumerate()
            .filter(|(_, c)| c.is_dirty)
        {
            chunk.is_dirty = false;

            let chunk_pos = glam::uvec3(
                index as u32 % self.num_chunks.x,
                (index as u32 / self.num_chunks.x) % self.num_chunks.y,
                index as u32 / (self.num_chunks.x * self.num_chunks.y),
            );

            let execution_time = self.distance_field_pipeline.compute(
                self.distance_fields[index].clone(),
                chunk,
                chunk_pos.as_vec3(),
            );

            self.ddf_generation_stats
                .df_execution_times
                .push(execution_time);

            self.push_delta_time = true;
        }
    }

    fn draw_frame(&mut self, renderer: &mut VulkanoWindowRenderer) {
        let before_pipeline_future =
            match renderer.acquire(Some(Duration::from_millis(1000)), |swapchain_image_views| {
                self.place_over_frame
                    .recreate_framebuffers(swapchain_image_views)
            }) {
                Err(e) => {
                    println!("{e}");
                    return;
                }
                Ok(future) => future,
            };

        let image = renderer.get_additional_image_view(0);

        let after_ray_march = self
            .ray_marcher_pipeline
            .compute(image.clone(), self.camera.into())
            .join(before_pipeline_future);

        if let Some(execution_time) = self.ray_marcher_pipeline.execution_time() {
            self.ddf_generation_stats
                .rm_execution_times
                .push(execution_time);
        }

        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            Window::new("Transparent Window")
                .anchor(Align2([Align::RIGHT, Align::TOP]), vec2(-545.0, 500.0))
                .resizable(false)
                .default_width(300.0)
                .frame(
                    Frame::NONE
                        .fill(Color32::from_white_alpha(125))
                        .shadow(Shadow {
                            spread: 8,
                            blur: 10,
                            color: Color32::from_black_alpha(125),
                            ..Default::default()
                        })
                        .corner_radius(CornerRadius::same(5))
                        .inner_margin(Margin::same(10)),
                )
                .show(&ctx, |ui| {
                    ui.colored_label(Color32::BLACK, "Content :)");
                });
        });

        let after_renderpass_future = self.place_over_frame.render(
            after_ray_march,
            image,
            renderer.swapchain_image_view(),
            renderer.image_index(),
            &mut self.gui,
        );

        renderer.present(after_renderpass_future, true);
    }
}

struct DDFGenerationStats {
    df_execution_times: Vec<f32>,
    rm_execution_times: Vec<f32>,
    frame_times: Vec<f64>,
}
