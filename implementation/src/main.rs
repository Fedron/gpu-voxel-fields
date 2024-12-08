#![feature(generic_const_exprs)]
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
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    memory::allocator::AllocationCreateInfo,
    sync::GpuFuture,
};
use vulkano_util::{context::VulkanoContext, renderer::VulkanoWindowRenderer};
use winit::{
    event::{DeviceEvent, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
};
use world::World;

mod app;
mod camera;
mod distance_field_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;
mod ray;
mod ray_marcher_pipeline;
mod world;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new()?;
    let mut app = App::<VoxelsApp>::new(&event_loop);

    println!(
        "\
Usage:
    Esc: Quit\
        ",
    );

    event_loop.run_app(&mut app)
}

struct VoxelsApp {
    distance_field_pipeline: DistanceFieldPipeline,
    ray_marcher_pipeline: RayMarcherPipeline,
    place_over_frame: RenderPassPlaceOverFrame,

    camera: Camera,
    camera_controller: CameraController,
    distance_field: Arc<ImageView>,
    world: World<8, 8, 8>,
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
                world.set(glam::uvec3(x, 0, z), world::Voxel::Stone);
            }
        }

        let distance_field = {
            let image = Image::new(
                context.memory_allocator().clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim3d,
                    format: Format::R8_UINT,
                    extent: world.size(),
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();

            ImageView::new_default(image).unwrap()
        };

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
                glam::Vec3::ZERO,
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
                if *key == KeyCode::Escape && state.is_pressed() {
                    event_loop.exit();
                }

                self.camera_controller.process_keyboard(*key, *state);
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
                            world::Voxel::Stone,
                        );
                    }
                } else if *button == MouseButton::Right && state.is_pressed() {
                    let hit = self
                        .world
                        .is_voxel_hit(Ray::new(self.camera.position, self.camera.front()));
                    if hit.does_intersect {
                        self.world
                            .set(hit.voxel_position.unwrap(), world::Voxel::Air);
                    }
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

        if self.world.is_dirty {
            let now = Instant::now();

            self.distance_field_pipeline
                .compute(self.distance_field.clone(), &self.world);
            self.world.is_dirty = false;

            let elapsed = now.elapsed();
            println!("Regenerated distance field in {:.2?}", elapsed);
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
