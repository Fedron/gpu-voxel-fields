use std::time::{Duration, Instant};

use vulkano::{
    image::ImageUsage,
    instance::{InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions},
    swapchain::PresentMode,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
};

/// Allows for state to be run as part of a winit + vulkan application.
pub trait AppState {
    const WINDOW_TITLE: &'static str;

    /// Initializes the state.
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        context: &VulkanoContext,
        window_renderer: &VulkanoWindowRenderer,
    ) -> Self;

    /// Handles `WindowEvent`s flushed by winit.
    fn handle_window_event(
        &mut self,
        window: &winit::window::Window,
        event_loop: &ActiveEventLoop,
        event: &WindowEvent,
    );
    /// Handles `DeviceEvent`s flushed by winit.
    fn handle_device_event(&mut self, event: &DeviceEvent);

    /// Updates the current state at the beginning of a `RedrawRequested` window event.
    fn update(&mut self, delta_time: Duration);
    /// Draws the state, run after [`update`].
    fn draw_frame(&mut self, renderer: &mut VulkanoWindowRenderer);
}

/// A Vulkan enabled winit application.
pub struct App<T>
where
    T: AppState + 'static,
{
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,

    pub frame_stats: FrameStats,
    pub state: Option<T>,
}

impl<T> App<T>
where
    T: AppState + 'static,
{
    /// Creates a new default Vulkano context and windows for use in a winit application.
    pub fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    ..InstanceExtensions::empty()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let windows = VulkanoWindows::default();

        App {
            context,
            windows,

            frame_stats: Default::default(),
            state: None,
        }
    }
}

impl<T> ApplicationHandler for App<T>
where
    T: AppState + 'static,
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _id = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: T::WINDOW_TITLE.to_string(),
                present_mode: PresentMode::Fifo,
                cursor_locked: true,
                ..Default::default()
            },
            |ci| {
                ci.image_format = vulkano::format::Format::B8G8R8A8_UNORM;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();

        window_renderer.add_additional_image_view(
            0,
            DEFAULT_IMAGE_FORMAT,
            ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
        );

        self.state = Some(T::new(event_loop, &self.context, &window_renderer));
        self.frame_stats = Default::default();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if self.state.is_none()
            || self
                .windows
                .get_primary_renderer_mut()
                .unwrap()
                .window()
                .inner_size()
                .width
                == 0
            || self
                .windows
                .get_primary_renderer_mut()
                .unwrap()
                .window()
                .inner_size()
                .height
                == 0
        {
            return;
        }

        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(_) => {
                self.windows.get_primary_renderer_mut().unwrap().resize();
                state.handle_window_event(
                    self.windows.get_window(window_id).unwrap(),
                    event_loop,
                    &event,
                );
            }
            WindowEvent::RedrawRequested => {
                self.frame_stats.update();
                state.update(self.frame_stats.delta_time);
                state.draw_frame(self.windows.get_primary_renderer_mut().unwrap());

                self.windows
                    .get_primary_renderer_mut()
                    .unwrap()
                    .window()
                    .set_title(&format!(
                        "{} - dt: {:.2}",
                        T::WINDOW_TITLE,
                        self.frame_stats.delta_time.as_secs_f64() * 1000.0
                    ));
            }
            ev => state.handle_window_event(
                self.windows.get_window(window_id).unwrap(),
                event_loop,
                &ev,
            ),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state {
            state.handle_device_event(&event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

/// Stores various information about frame timing.
#[derive(Debug, Clone)]
pub struct FrameStats {
    pub _start_time: Instant,
    pub delta_time: Duration,
    pub last_frame_time: Instant,

    pub frame_times: Vec<f64>,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self {
            _start_time: Instant::now(),
            delta_time: Default::default(),
            last_frame_time: Instant::now(),
            frame_times: Default::default(),
        }
    }
}

impl FrameStats {
    /// Updates the frame stats since the last time they were updated.
    pub fn update(&mut self) {
        self.delta_time = self.last_frame_time.elapsed();
        self.last_frame_time = Instant::now();

        self.frame_times
            .push(self.delta_time.as_secs_f64() * 1000.0);
    }
}
