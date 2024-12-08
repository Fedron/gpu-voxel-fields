#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use camera::Camera;
use distance_field_pipeline::DistanceFieldPipeline;
use input::InputState;
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
    swapchain::PresentMode,
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::DEFAULT_IMAGE_FORMAT,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};
use world::World;

mod camera;
mod distance_field_pipeline;
mod input;
mod pixels_draw_pipeline;
mod place_over_frame;
mod ray;
mod ray_marcher_pipeline;
mod world;

fn main() -> Result<(), impl Error> {
    // Create the event loop.
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    println!(
        "\
Usage:
    Esc: Quit\
        ",
    );

    event_loop.run_app(&mut app)
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    distance_field_pipeline: DistanceFieldPipeline,
    ray_marcher_pipeline: RayMarcherPipeline,
    distance_field: Arc<ImageView>,
    camera: Camera,
    world: World<8, 8, 8>,
    /// Our render pipeline (pass).
    place_over_frame: RenderPassPlaceOverFrame,
    /// Time tracking, useful for frame independent movement.
    time: Instant,
    dt: f32,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
    /// Input state to handle mouse positions, continuous movement etc.
    input_state: InputState,
    render_target_id: usize,
}

impl App {
    fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());
        let windows = VulkanoWindows::default();

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

        App {
            context,
            windows,
            descriptor_set_allocator,
            command_buffer_allocator,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let _id = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "Voxels".to_string(),
                present_mode: PresentMode::Fifo,
                cursor_locked: true,
                cursor_visible: false,
                ..Default::default()
            },
            |_| {},
        );

        // Add our render target image onto which we'll be rendering our fractals.
        let render_target_id = 0;
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();

        // Make sure the image usage is correct (based on your pipeline).
        window_renderer.add_additional_image_view(
            render_target_id,
            DEFAULT_IMAGE_FORMAT,
            ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
        );

        let gfx_queue = self.context.graphics_queue();

        let mut world = World::new(self.context.memory_allocator().clone());
        for x in 0..world.size()[0] {
            for z in 0..world.size()[2] {
                world.set(glam::uvec3(x, 0, z), world::Voxel::Stone);
            }
        }

        let distance_field = {
            let image = Image::new(
                self.context.memory_allocator().clone(),
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
            gfx_queue.clone(),
            self.context.memory_allocator().clone(),
            self.command_buffer_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            &world,
        );

        self.rcx = Some(RenderContext {
            render_target_id,
            distance_field_pipeline: DistanceFieldPipeline::new(
                gfx_queue.clone(),
                self.context.memory_allocator().clone(),
                self.command_buffer_allocator.clone(),
                self.descriptor_set_allocator.clone(),
            ),
            ray_marcher_pipeline,
            distance_field,
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
            world,
            place_over_frame: RenderPassPlaceOverFrame::new(
                gfx_queue.clone(),
                self.command_buffer_allocator.clone(),
                self.descriptor_set_allocator.clone(),
                window_renderer.swapchain_format(),
                window_renderer.swapchain_image_views(),
            ),
            time: Instant::now(),
            dt: 0.0,
            dt_sum: 0.0,
            frame_count: 0.0,
            avg_fps: 0.0,
            input_state: InputState::new(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();
        let window_size = renderer.window().inner_size();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                renderer.resize();
                rcx.camera.resize((new_size.width, new_size.height));
            }
            WindowEvent::RedrawRequested => {
                // Skip this frame when minimized.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.update_state_after_inputs();

                // Start the frame.
                let before_pipeline_future = match renderer.acquire(
                    Some(Duration::from_millis(1000)),
                    |swapchain_image_views| {
                        rcx.place_over_frame
                            .recreate_framebuffers(swapchain_image_views)
                    },
                ) {
                    Err(e) => {
                        println!("{e}");
                        return;
                    }
                    Ok(future) => future,
                };

                // Retrieve the target image.
                let image = renderer.get_additional_image_view(rcx.render_target_id);

                // Compute our fractal (writes to target image). Join future with
                // `before_pipeline_future`.
                let after_ray_march = rcx
                    .ray_marcher_pipeline
                    .compute(image.clone(), rcx.distance_field.clone(), rcx.camera.into())
                    .join(before_pipeline_future);

                // Render the image over the swapchain image, inputting the previous future.
                let after_renderpass_future = rcx.place_over_frame.render(
                    after_ray_march,
                    image,
                    renderer.swapchain_image_view(),
                    renderer.image_index(),
                );

                // Finish the frame (which presents the view), inputting the last future. Wait for
                // the future so resources are not in use when we render.
                renderer.present(after_renderpass_future, true);

                rcx.input_state.reset();
                rcx.update_time();
                renderer.window().set_title(&format!(
                    "fps: {:.2} dt: {:.2}",
                    rcx.avg_fps(),
                    rcx.dt()
                ));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state.is_pressed() {
                    let hit = rcx
                        .world
                        .is_voxel_hit(Ray::new(rcx.camera.position, rcx.camera.front()));
                    if hit.does_intersect {
                        rcx.world.set(
                            hit.voxel_position
                                .unwrap()
                                .saturating_add_signed(hit.face_normal.unwrap()),
                            world::Voxel::Stone,
                        );
                    }
                } else if button == MouseButton::Right && state.is_pressed() {
                    let hit = rcx
                        .world
                        .is_voxel_hit(Ray::new(rcx.camera.position, rcx.camera.front()));
                    if hit.does_intersect {
                        rcx.world
                            .set(hit.voxel_position.unwrap(), world::Voxel::Air);
                    }
                }
            }
            _ => {
                // Pass event for the app to handle our inputs.
                rcx.input_state.handle_input(window_size, &event);
            }
        }

        if rcx.input_state.should_quit {
            event_loop.exit();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();
        match event {
            DeviceEvent::MouseMotion { delta } => {
                rcx.camera.on_mouse_motion(delta);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

impl RenderContext {
    /// Updates app state based on input state.
    fn update_state_after_inputs(&mut self) {
        self.camera.handle_input(&self.input_state, self.dt);

        if self.world.is_dirty {
            let now = Instant::now();

            self.distance_field_pipeline
                .compute(self.distance_field.clone(), &self.world);
            self.world.is_dirty = false;

            let elapsed = now.elapsed();
            println!("Regenerated distance field in {:.2?}", elapsed);
        }
    }

    /// Returns the average FPS.
    fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Returns the delta time in milliseconds.
    fn dt(&self) -> f32 {
        self.dt * 1000.0
    }

    /// Updates times and dt at the end of each frame.
    fn update_time(&mut self) {
        // Each second, update average fps & reset frame count & dt sum.
        if self.dt_sum > 1.0 {
            self.avg_fps = self.frame_count / self.dt_sum;
            self.frame_count = 0.0;
            self.dt_sum = 0.0;
        }
        self.dt = self.time.elapsed().as_secs_f32();
        self.dt_sum += self.dt;
        self.frame_count += 1.0;
        self.time = Instant::now();
    }
}
