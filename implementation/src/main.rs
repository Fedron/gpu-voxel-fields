#![feature(generic_const_exprs)]
#![feature(duration_millis_float)]
#![feature(map_try_insert)]
#![allow(incomplete_features)]

use app::{App, AppState};
use camera::{Camera, CameraController};
use distance_field_pipeline::DistanceFieldPipeline;
use place_over_frame::RenderPassPlaceOverFrame;
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
    event::{DeviceEvent, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};
use world::{voxel::Voxel, World};

mod app;
mod camera;
mod distance_field_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;
mod ray;
mod ray_marcher_pipeline;
mod utils;
mod world;

const STEPS_PER_SECOND: u64 = 10;
const ENABLE_WORLD_UPDATES: bool = true;
const WORLD_SIZE: usize = 32;
const CHUNK_SIZE: usize = 16;

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
    distance_fields: Vec<Subbuffer<[u32]>>,
    world: World,
    generation_times: Vec<f32>,
    last_update: Instant,
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

        let num_chunks = glam::UVec3::splat(WORLD_SIZE as u32)
            .saturating_div(glam::UVec3::splat(CHUNK_SIZE as u32));
        let mut world = World::new(
            glam::UVec3::splat(CHUNK_SIZE as u32),
            num_chunks,
            context.memory_allocator().clone(),
        );
        for x in 0..world.size().x {
            for z in 0..world.size().z {
                world.set(glam::uvec3(x, 0, z), Voxel::Stone);
            }
        }

        world.set(
            glam::uvec3(
                (world.size().x as f32 * 0.25) as u32,
                (world.size().y as f32 * 0.75) as u32,
                (world.size().z as f32 * 0.25) as u32,
            ),
            Voxel::SandGenerator,
        );

        world.set(
            glam::uvec3(
                (world.size().x as f32 * 0.75) as u32,
                (world.size().y as f32 * 0.75) as u32,
                (world.size().z as f32 * 0.75) as u32,
            ),
            Voxel::WaterGenerator,
        );
        world.update_count = 1;

        let mut distance_fields = Vec::with_capacity(num_chunks.element_product() as usize);
        for _ in 0..num_chunks.element_product() {
            distance_fields.push(
                Buffer::new_slice::<u32>(
                    context.memory_allocator().clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                        ..Default::default()
                    },
                    glam::UVec3::splat(CHUNK_SIZE as u32).element_product() as u64,
                )
                .unwrap(),
            );
        }

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
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                window_renderer.swapchain_format(),
                window_renderer.swapchain_image_views(),
            ),

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
            distance_fields,
            world,
            generation_times: Vec::new(),
            last_update: Instant::now(),
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
                    _ => {}
                }
            }
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

        if ENABLE_WORLD_UPDATES
            && self.last_update.elapsed()
                > Duration::from_millis((1000.0 / STEPS_PER_SECOND as f32) as u64)
        {
            self.world.update();
            self.last_update = Instant::now();
        }

        for (index, chunk) in self
            .world
            .chunks
            .iter_mut()
            .enumerate()
            .filter(|(_, c)| c.is_dirty)
        {
            self.distance_field_pipeline
                .compute(self.distance_fields[index].clone(), chunk);
            chunk.is_dirty = false;

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
                self.distance_fields.clone(),
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
