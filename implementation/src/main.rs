#![feature(generic_const_exprs)]
#![feature(duration_millis_float)]
#![feature(map_try_insert)]
#![feature(variant_count)]
#![allow(incomplete_features)]

use app::{App, AppState};
use camera::{Camera, CameraController};
use distance_field::{brute_force::BruteForcePipeline, hybrid::HybridPipeline, DistanceFieldPipeline};
use egui::Color32;
use egui_plot::{Legend, Line, Plot};
use egui_winit_vulkano::{Gui, GuiConfig};
use place_over_frame::RenderPassPlaceOverFrame;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use ray::Ray;
use ray_marcher_pipeline::RayMarcherPipeline;
use strum::IntoEnumIterator;
use std::{
    error::Error,
    sync::Arc,
    time::{Duration, Instant},
};
use utils::{get_sphere_positions, Statistics};
use vulkano::{
    buffer::{allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}, BufferUsage, Subbuffer},
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
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

    Left ALT: Show/enable mouse
    Left SHIFT: Increase movement speed
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

    Ok(())
}

#[derive(Clone, Copy)]
struct WorldConfiguration {
    world_size: usize,
    chunk_size: usize,
    df_algorithm: distance_field::Algorithm,
    focal_size: usize,
}

#[derive(Clone)]
struct TestConfiguration {
    enabled: bool,
    modification_interval: u64,
    seed: u64,
    rng: SmallRng,
}

struct VoxelsApp {
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    distance_field_allocator: Arc<SubbufferAllocator>,
    compute_queue: Arc<Queue>,

    distance_field_pipeline: Box<dyn DistanceFieldPipeline>,
    ray_marcher_pipeline: RayMarcherPipeline,
    place_over_frame: RenderPassPlaceOverFrame,
    gui: Gui,

    camera: Camera,
    camera_controller: CameraController,
    world: World,
    distance_fields: Vec<Subbuffer<[u8]>>,

    ddf_generation_stats: DDFGenerationStats,
    last_modification: Instant,

    voxel_to_place: Voxel,
    lmb_held: bool,
    rmb_held: bool,
    brush_size: u32,
    alt_held: bool,

    world_configuration: WorldConfiguration,
    test_configuration: TestConfiguration,
}

impl VoxelsApp {
    fn create_world(
        config: WorldConfiguration,
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        distance_field_allocator: Arc<SubbufferAllocator>,
    ) -> (
        World,
        Vec<Subbuffer<[u8]>>,
        Box<dyn DistanceFieldPipeline>,
        RayMarcherPipeline,
    ) {
        let num_chunks = glam::UVec3::splat(config.world_size as u32)
            .saturating_div(glam::UVec3::splat(config.chunk_size as u32));
        let mut world = World::new(
            glam::UVec3::splat(config.chunk_size as u32),
            num_chunks,
            memory_allocator.clone(),
        );

        for x in 0..world.size().x {
            for z in 0..world.size().z {
                world.set(glam::uvec3(x, 0, z), Voxel::Grey);
            }
        }
        world.update_count = 0;

        let distance_field_pipeline: Box<dyn DistanceFieldPipeline> = match config.df_algorithm {
            distance_field::Algorithm::BruteForce => Box::new(BruteForcePipeline::new(queue.clone(), command_buffer_allocator.clone(), descriptor_set_allocator.clone(), config.chunk_size as u32)),
            distance_field::Algorithm::FastIterative => Box::new(HybridPipeline::new(
                queue.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                distance_field_allocator.clone(),
                config.chunk_size as u32,
                glam::Vec3::ZERO,
                world.num_chunks.as_vec3(),
            )),
            distance_field::Algorithm::JumpFlooding => Box::new(HybridPipeline::new(
                queue.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                distance_field_allocator.clone(),
                config.chunk_size as u32,
                glam::Vec3::NEG_ONE,
                glam::Vec3::NEG_ONE,
            )),
            distance_field::Algorithm::Hybrid => {
                let focal_point = (num_chunks / 2).saturating_sub(glam::UVec3::ONE);
                let focus_size = config.focal_size as u32;

                Box::new(HybridPipeline::new(
                    queue.clone(),
                    memory_allocator.clone(),
                    command_buffer_allocator.clone(),
                    descriptor_set_allocator.clone(),
                    distance_field_allocator.clone(),
                    config.chunk_size as u32,
                    focal_point
                    .saturating_sub(glam::UVec3::splat(focus_size))
                    .as_vec3(),
                    (focal_point + focus_size).as_vec3(),
                )
            )},
        };

        let mut distance_fields = Vec::with_capacity(num_chunks.element_product() as usize);
        for _ in 0..num_chunks.element_product() {
            let buffer = distance_field_allocator.allocate(distance_field_pipeline.layout()).unwrap();
            distance_fields.push(buffer);
        }

        let ray_marcher_pipeline = RayMarcherPipeline::new(
            queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            distance_fields.clone(),
            &world,
        );

        (
            world,
            distance_fields,
            distance_field_pipeline,
            ray_marcher_pipeline,
        )
    }
}

impl AppState for VoxelsApp {
    const WINDOW_TITLE: &'static str = "Voxels";

    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        context: &VulkanoContext,
        window_renderer: &VulkanoWindowRenderer,
    ) -> Self {
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

        let world_configuration = WorldConfiguration {
            world_size: 128,
            chunk_size: 32,
            df_algorithm: distance_field::Algorithm::Hybrid,
            focal_size: 2,
        };
        let test_configuration = TestConfiguration {
            enabled: false,
            modification_interval: 200,
            seed: 6683787,
            rng: SmallRng::seed_from_u64(6683787),
        };

        let distance_field_allocator = Arc::new(SubbufferAllocator::new(
            context.memory_allocator().clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        ));

        let (world, distance_fields, distance_field_pipeline, ray_marcher_pipeline) =
            VoxelsApp::create_world(
                world_configuration,
                context.graphics_queue().clone(),
                context.memory_allocator().clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                distance_field_allocator.clone(),
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
            memory_allocator: context.memory_allocator().clone(),
            command_buffer_allocator,
            descriptor_set_allocator,
            distance_field_allocator,
            compute_queue: context.graphics_queue().clone(),

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
            last_modification: Instant::now(),

            voxel_to_place: Voxel::Grey,
            lmb_held: false,
            rmb_held: false,
            brush_size: 1,
            alt_held: false,

            world_configuration,
            test_configuration,
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
        if self.test_configuration.enabled
            && self.last_modification.elapsed().as_millis() as u64
                > self.test_configuration.modification_interval
        {
            self.ddf_generation_stats
                .frame_times
                .push(delta_time.as_secs_f64() * 1000.0);

            let positions = get_sphere_positions(
                glam::ivec3(
                    self.test_configuration
                        .rng
                        .gen_range(0..self.world.size().x) as i32,
                    self.test_configuration
                        .rng
                        .gen_range(0..self.world.size().y) as i32,
                    self.test_configuration
                        .rng
                        .gen_range(0..self.world.size().z) as i32,
                ),
                self.test_configuration
                    .rng
                    .gen_range(1..(self.world_configuration.chunk_size as u32)),
            );
            for &position in positions.iter() {
                if self.world.is_in_bounds_ivec3(position) {
                    self.world.set(
                        position.as_uvec3(),
                        (self
                            .test_configuration
                            .rng
                            .gen_range(0..std::mem::variant_count::<Voxel>())
                            as u32)
                            .into(),
                    );
                }
            }
            self.last_modification = Instant::now();
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
                index as u32 % self.world.num_chunks.x,
                (index as u32 / self.world.num_chunks.x) % self.world.num_chunks.y,
                index as u32 / (self.world.num_chunks.x * self.world.num_chunks.y),
            );

            self.distance_field_pipeline.compute(
                self.distance_fields[index].clone(),
                chunk,
                chunk_pos.as_vec3(),
            );

            self.ddf_generation_stats
                .df_execution_times
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
            .compute(image.clone(), self.camera.into())
            .join(before_pipeline_future);

        if self.test_configuration.enabled {
            if let Some(execution_time) = self.ray_marcher_pipeline.execution_time() {
                self.ddf_generation_stats
                    .rm_execution_times
                    .push(execution_time);
            }
        }

        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            egui::SidePanel::right("main_right_panel").show(&ctx, |ui| {
                ui.label("Voxel Painter");

                egui::CollapsingHeader::new("Brush Settings").default_open(true).show(ui, |ui| {
                    ui.label("Configure your brush.");
                    ui.add_space(10.0);

                    ui.add(egui::Slider::new(&mut self.brush_size, 1..=(self.world_configuration.chunk_size as u32)).text("Brush Size"));
                
                    egui::ComboBox::from_label("Active Voxel")
                        .selected_text(self.voxel_to_place.to_string())
                        .show_ui(ui, |ui| {
                            for voxel in Voxel::iter() {
                                ui.selectable_value(&mut self.voxel_to_place, voxel, voxel.to_string());
                            }
                        });
                });

                egui::CollapsingHeader::new("World Settings").show(ui, |ui| {
                    ui.label("Configure the world generation settings.");
                    ui.add_space(10.0);

                    egui::ComboBox::from_label("World Size")
                        .selected_text(format!("{} voxels", self.world_configuration.world_size))
                        .show_ui(ui, |ui| {
                            let start = self.world_configuration.chunk_size.ilog2();
                            for i in start..=10 {
                                let value = 2_usize.pow(i);
                                ui.selectable_value(
                                    &mut self.world_configuration.world_size,
                                    value,
                                    format!("{}", value),
                                );
                            }
                        });

                    egui::ComboBox::from_label("Chunk Size")
                        .selected_text(format!("{} voxels", self.world_configuration.chunk_size))
                        .show_ui(ui, |ui| {
                            for i in 2..=10 {
                                let value = 2_usize.pow(i);
                                ui.selectable_value(
                                    &mut self.world_configuration.chunk_size,
                                    value,
                                    format!("{}", value),
                                );
                            }
                        });

                    egui::CollapsingHeader::new("Rendering Settings").show(ui, |ui| {
                        ui.label("Configure the world rendering settings");
                        ui.add_space(10.0);

                        egui::ComboBox::from_label("Algorithm")
                            .selected_text(self.world_configuration.df_algorithm.to_string())
                            .show_ui(ui, |ui| {
                                for algorithm in distance_field::Algorithm::iter() {
                                    ui.selectable_value(&mut self.world_configuration.df_algorithm, algorithm, algorithm.to_string());
                                }
                            });

                        ui.colored_label(Color32::RED, "In order for this to take effect the world must be regenerated.")
                    });

                    ui.add_space(10.0);
                    if ui.add(egui::Button::new("Regenerate")).clicked() {
                        let (world, distance_fields, distance_field_pipeline, ray_marcher_pipeline) =
                            VoxelsApp::create_world(
                                self.world_configuration,
                                self.compute_queue.clone(),
                                self.memory_allocator.clone(),
                                self.command_buffer_allocator.clone(),
                                self.descriptor_set_allocator.clone(),
                                self.distance_field_allocator.clone(),
                            );

                        self.world = world;
                        self.distance_fields = distance_fields;
                        self.distance_field_pipeline = distance_field_pipeline;
                        self.ray_marcher_pipeline = ray_marcher_pipeline;

                        self.ddf_generation_stats.frame_times.clear();
                    }
                });

                egui::CollapsingHeader::new("Test Settings").show(ui, |ui| {
                    ui.label("A test mode that will randomly place spheres in the world to test performance.");
                    ui.add_space(10.0);

                    ui.add(egui::Slider::new(&mut self.test_configuration.seed, 0..=std::u64::MAX).text("Seed"));
                    ui.add(egui::Slider::new(&mut self.test_configuration.modification_interval, 1..=10000).text("Modification Interval"));

                    ui.add_space(10.0);
                    if ui.add(egui::Button::new(if self.test_configuration.enabled {
                        "Stop"
                    } else {
                        "Start"
                    })).clicked() {
                        self.test_configuration.enabled = !self.test_configuration.enabled;
                        if self.test_configuration.enabled {
                            self.ddf_generation_stats.df_execution_times.clear();
                            self.ddf_generation_stats.frame_times.clear();
                            self.ddf_generation_stats.rm_execution_times.clear();
                            self.world.update_count = 0;
                        }
                    }

                    if !self.test_configuration.enabled && !self.ddf_generation_stats.frame_times.is_empty() {
                        ui.separator();
                        ui.label("Results");

                        let avg_dt =
                            self.ddf_generation_stats.frame_times.iter().sum::<f64>() / self.ddf_generation_stats.frame_times.len() as f64;
                        ui.label(format!("Average Delta Time: {:.5}ms", avg_dt));
                        ui.label(format!("Average FPS: {:.5}", 1000.0 / avg_dt));

                        ui.add_space(5.0);
                        ui.label(format!("{} world updates", self.world.update_count));

                        let stats = Statistics::calculate(&self.ddf_generation_stats.df_execution_times);
                        ui.label(format!("{} distance field updates", self.ddf_generation_stats.df_execution_times.len()));

                        ui.add_space(5.0);
                        ui.label(format!("Average distance field update time {:.5}ms", stats.mean));

                        let stats = Statistics::calculate(&self.ddf_generation_stats.rm_execution_times);
                        ui.label(format!("Average ray marching time {:.5}ms", stats.mean));
                    }
                });
            });

            if self.test_configuration.enabled || !self.ddf_generation_stats.frame_times.is_empty() {
                egui::TopBottomPanel::bottom("performance_panel").default_height(200.0).resizable(true).show(&ctx, |ui| {
                    ui.label("Performance Statistics (Real-time)");
                    ui.add_space(5.0);
                    Plot::new("Performance").legend(Legend::default().position(egui_plot::Corner::LeftTop)).show(ui, |ui| {
                        ui.line(Line::new("Frame Times", self.ddf_generation_stats.frame_times[self.ddf_generation_stats.frame_times.len().saturating_sub(25)..].iter().enumerate().map(|(i, v)| [i as f64, *v]).collect::<Vec<_>>()));
                        ui.line(Line::new("Ray Marcher Execution Time", self.ddf_generation_stats.rm_execution_times[self.ddf_generation_stats.rm_execution_times.len().saturating_sub(25)..].iter().enumerate().map(|(i, v)| [i as f64, *v as f64]).collect::<Vec<_>>()));
                        ui.line(Line::new("Distance Field Exceution Time", self.ddf_generation_stats.df_execution_times[self.ddf_generation_stats.df_execution_times.len().saturating_sub(25)..].iter().enumerate().map(|(i, v)| [i as f64, *v as f64]).collect::<Vec<_>>()));
                    });
                });
            }
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
