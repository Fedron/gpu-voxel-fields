#![feature(generic_const_exprs)]
#![feature(duration_millis_float)]
#![feature(map_try_insert)]
#![allow(incomplete_features)]

use app::{App, AppState};
use camera::{Camera, CameraController};
use distance_field_pipeline::DistanceFieldPipeline;
use place_over_frame::RenderPassPlaceOverFrame;
use ray::Ray;
use ray_marcher_pipeline::RayMarcherPipeline;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use utils::Statistics;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};
use vulkano_util::{context::VulkanoContext, renderer::VulkanoWindowRenderer};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};
use world::{Voxel, World};

mod app;
mod camera;
mod crosshair_pipeline;
mod distance_field_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;
mod ray;
mod ray_marcher_pipeline;
mod utils;
mod world;

const STEPS_PER_SECOND: u64 = 10;
const ENABLE_WORLD_UPDATES: bool = true;
const WORLD_SIZE: usize = 8;

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::<VoxelsApp>::new(&event_loop);

    println!("Using {}\n", app.context.device_name());

    println!("Enable World Updates: {}", ENABLE_WORLD_UPDATES);
    println!("Steps per second: {}", STEPS_PER_SECOND);
    println!("World Size: {}x{}x{}\n", WORLD_SIZE, WORLD_SIZE, WORLD_SIZE);

    println!(
        "\
Fast GPU Generation of Signed Distance Fields from a Voxel Grid.

Voxel world rendered using a dynamically updated discrete distance field. Simple falling sand simulation to demo and
benchmark performance of DDF generation.

Generator voxels will continually spawn their respective voxel type. A sand and water generator voxel have been placed
in the top corner's of the world along with a flat stone voxel platform at the base of the world.

Usage:
    Esc: Quit

    LMB: Place active voxel
    RMB: Delete voxel

    1: Set 'Stone' as active voxel
    2: Set 'Sand' as active voxel
    3: Set 'Water as active voxel
    4: Set 'Sand Generator' as active voxel
    5: Set 'Water Generator' as active voxel\
        \n",
    );

    let now = Instant::now();

    event_loop.run_app(&mut app)?;

    println!(
        "Average FPS: {:.5}\nAverage Delta Time: {:.5}\n",
        app.frame_stats.fps_counts.iter().sum::<f32>() / app.frame_stats.fps_counts.len() as f32,
        app.frame_stats.dt_counts.iter().sum::<f32>() / app.frame_stats.dt_counts.len() as f32
    );

    let state = app.state.unwrap();
    let stats = Statistics::calculate(&state.generation_times);
    println!(
        "DDF Algorithm Runtime Statistics (ms) ({} entries)\n{:#?}\n",
        &state.generation_times.len(),
        stats
    );

    println!(
        "Average world updates per second: {:.5}/s\n",
        state.world.update_count as f64 / now.elapsed().as_secs_f64()
    );

    println!(
        "DDF Regeneration Events: {}\nWorld Updates: {}",
        state.generation_times.len(),
        state.world.update_count
    );

    Ok(())
}

struct VoxelsApp {
    distance_field_pipeline: DistanceFieldPipeline,
    ray_marcher_pipeline: RayMarcherPipeline,
    place_over_frame: RenderPassPlaceOverFrame,

    camera: Camera,
    camera_controller: CameraController,
    distance_field: Subbuffer<[u32]>,
    world: World<WORLD_SIZE, WORLD_SIZE, WORLD_SIZE>,
    generation_times: Vec<f32>,
    last_update: Instant,

    voxel_to_place: Voxel,
    lmb_held: bool,
    rmb_held: bool,
}

impl AppState for VoxelsApp {
    const WINDOW_TITLE: &'static str = "Voxels";

    fn new(context: &VulkanoContext, window_renderer: &VulkanoWindowRenderer) -> Self {
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

        let mut world = World::new(context.memory_allocator().clone());
        for x in 0..world.size()[0] {
            for z in 0..world.size()[2] {
                world.set(glam::uvec3(x, 0, z), Voxel::Stone);
            }
        }

        world.set(
            glam::uvec3(
                (world.size()[0] as f32 * 0.25) as u32,
                (world.size()[1] as f32 * 0.75) as u32,
                (world.size()[2] as f32 * 0.25) as u32,
            ),
            Voxel::SandGenerator,
        );

        world.set(
            glam::uvec3(
                (world.size()[0] as f32 * 0.75) as u32,
                (world.size()[1] as f32 * 0.75) as u32,
                (world.size()[2] as f32 * 0.75) as u32,
            ),
            Voxel::WaterGenerator,
        );
        world.update_count = 1;

        let distance_field = Buffer::new_slice::<u32>(
            context.memory_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            world.size()[0] as u64 * world.size()[1] as u64 * world.size()[2] as u64,
        )
        .unwrap();

        let ray_marcher_pipeline = RayMarcherPipeline::new(
            context.graphics_queue().clone(),
            context.memory_allocator().clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            &world,
        );

        Self {
            distance_field_pipeline: DistanceFieldPipeline::new(
                context.graphics_queue().clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
            ),
            ray_marcher_pipeline,
            place_over_frame: RenderPassPlaceOverFrame::new(
                context.graphics_queue().clone(),
                context.memory_allocator().clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                window_renderer.swapchain_format(),
                window_renderer.swapchain_image_views(),
            ),

            camera: Camera::new(
                glam::vec3(
                    -(world.size()[0] as f32),
                    world.size()[1] as f32 / 2.0,
                    world.size()[2] as f32 / 2.0,
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
            distance_field,
            world,
            generation_times: Vec::new(),
            last_update: Instant::now(),

            voxel_to_place: Voxel::Stone,
            lmb_held: false,
            rmb_held: false,
        }
    }

    fn handle_window_event(&mut self, event_loop: &ActiveEventLoop, event: &WindowEvent) {
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
                    KeyCode::Digit4 => self.voxel_to_place = Voxel::SandGenerator,
                    KeyCode::Digit5 => self.voxel_to_place = Voxel::WaterGenerator,
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
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => self
                .camera_controller
                .process_mouse(delta.0 as f32, delta.1 as f32),
            _ => {}
        }
    }

    fn update(&mut self, delta_time: Duration) {
        self.camera_controller
            .update_camera(&mut self.camera, delta_time.as_secs_f32());

        if self.lmb_held {
            let hit = self
                .world
                .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
            if hit.does_intersect {
                self.world.set(
                    hit.voxel_position
                        .unwrap()
                        .saturating_add_signed(hit.face_normal.unwrap()),
                    self.voxel_to_place,
                );
            }
        } else if self.rmb_held {
            let hit = self
                .world
                .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
            if hit.does_intersect {
                self.world.set(hit.voxel_position.unwrap(), Voxel::Air);
            }
        }

        if ENABLE_WORLD_UPDATES
            && self.last_update.elapsed()
                > Duration::from_millis((1000.0 / STEPS_PER_SECOND as f32) as u64)
        {
            self.world.update();
            self.last_update = Instant::now();
        }

        if self.world.is_dirty {
            self.distance_field_pipeline
                .compute(self.distance_field.clone(), &self.world);
            self.world.is_dirty = false;

            self.generation_times
                .push(self.distance_field_pipeline.execution_time());
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
            .compute(
                image.clone(),
                self.distance_field.clone(),
                self.camera.into(),
            )
            .join(before_pipeline_future);

        let after_renderpass_future = self.place_over_frame.render(
            after_ray_march,
            image,
            renderer.swapchain_image_view(),
            renderer.image_index(),
        );

        renderer.present(after_renderpass_future, true);
    }
}
