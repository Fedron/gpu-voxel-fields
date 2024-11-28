use std::{
    num::NonZeroU32,
    time::{Duration, Instant},
};

use glium::{
    glutin::{
        config::ConfigTemplateBuilder,
        context::{ContextApi, ContextAttributesBuilder},
        display::GetGlDisplay,
        prelude::{GlDisplay, NotCurrentGlContext},
        surface::{SurfaceAttributesBuilder, WindowSurface},
    },
    Display,
};
use glutin_winit::DisplayBuilder;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    raw_window_handle::HasWindowHandle,
    window::{CursorGrabMode, Window, WindowId},
};

pub trait ApplicationContext {
    const WINDOW_TITLE: &'static str;

    fn new(display: &Display<WindowSurface>) -> Self;
    fn handle_window_event(
        &mut self,
        display: &Display<WindowSurface>,
        window: &Window,
        event: &WindowEvent,
    );
    fn handle_device_event(&mut self, event: &DeviceEvent);
    fn update(&mut self, delta_time: Duration);
    fn draw_frame(&mut self, display: &Display<WindowSurface>);
}

pub struct State<T> {
    pub display: Display<WindowSurface>,
    pub window: Window,
    pub context: T,

    last_frame_time: Instant,
    delta_time: Duration,
}

struct App<T> {
    state: Option<State<T>>,
    visible: bool,
    close_promptly: bool,
}

impl<T: ApplicationContext + 'static> ApplicationHandler<()> for App<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.state = Some(State::new(event_loop, self.visible));
        if !self.visible && self.close_promptly {
            event_loop.exit();
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.state = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(new_size) => {
                if let Some(state) = &mut self.state {
                    state.display.resize(new_size.into());
                    state
                        .context
                        .handle_window_event(&state.display, &state.window, &event);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    let current_time = Instant::now();
                    state.delta_time = current_time.duration_since(state.last_frame_time);
                    state.last_frame_time = current_time;

                    state.context.update(state.delta_time);
                    state.context.draw_frame(&state.display);
                    if self.close_promptly {
                        event_loop.exit();
                    }
                }
            }
            glium::winit::event::WindowEvent::CloseRequested
            | glium::winit::event::WindowEvent::KeyboardInput {
                event:
                    glium::winit::event::KeyEvent {
                        state: glium::winit::event::ElementState::Pressed,
                        logical_key:
                            glium::winit::keyboard::Key::Named(glium::winit::keyboard::NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            ev => {
                if let Some(state) = &mut self.state {
                    state
                        .context
                        .handle_window_event(&state.display, &state.window, &ev);
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = &mut self.state {
            state.context.handle_device_event(&event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

impl<T: ApplicationContext + 'static> State<T> {
    pub fn new(event_loop: &ActiveEventLoop, visible: bool) -> Self {
        let window_attributes = Window::default_attributes()
            .with_title(T::WINDOW_TITLE)
            .with_visible(visible)
            .with_inner_size::<PhysicalSize<u32>>((1920_u32, 1080_u32).into());
        let config_template_builder = ConfigTemplateBuilder::new();
        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_attributes));

        let (window, gl_config) = display_builder
            .build(event_loop, config_template_builder, |mut configs| {
                configs.next().unwrap()
            })
            .unwrap();
        let window = window.unwrap();
        window
            .set_cursor_grab(CursorGrabMode::Confined)
            .unwrap_or_else(|_| {
                window
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .expect("to lock cursor to window as a fallback")
            });
        window.set_cursor_visible(false);

        let window_handle = window.window_handle().expect("to obtain a window handle");
        let context_attributes = ContextAttributesBuilder::new().build(Some(window_handle.into()));
        let fallback_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(Some(window_handle.into()));

        let not_current_gl_context = Some(unsafe {
            gl_config
                .display()
                .create_context(&gl_config, &context_attributes)
                .unwrap_or_else(|_| {
                    gl_config
                        .display()
                        .create_context(&gl_config, &fallback_context_attributes)
                        .expect("to create fallback ES context")
                })
        });

        let (width, height): (u32, u32) = if visible {
            window.inner_size().into()
        } else {
            (800, 600)
        };
        let attrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            window_handle.into(),
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
        );
        let surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &attrs)
                .unwrap()
        };
        let current_context = not_current_gl_context
            .unwrap()
            .make_current(&surface)
            .unwrap();
        let display = Display::from_context_surface(current_context, surface).unwrap();

        Self::from_display_window(display, window)
    }

    pub fn from_display_window(display: Display<WindowSurface>, window: Window) -> Self {
        let context = T::new(&display);
        Self {
            display,
            window,
            context,

            last_frame_time: Instant::now(),
            delta_time: Duration::ZERO,
        }
    }

    pub fn run_loop() {
        let event_loop = EventLoop::builder().build().expect("to build event loop");
        let mut app = App::<T> {
            state: None,
            visible: true,
            close_promptly: false,
        };
        let result = event_loop.run_app(&mut app);
        result.unwrap();
    }
}
