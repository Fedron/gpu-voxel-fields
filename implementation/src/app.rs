use std::{
    rc::Rc,
    time::{Duration, Instant},
};

use glium::{glutin::surface::WindowSurface, Display, Surface};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    platform::pump_events::EventLoopExtPumpEvents,
};

/// Wrapper for a winit window and a glium display.
pub struct Window {
    /// Underlying winit window.
    pub winit: winit::window::Window,
    /// Underlying glium display.
    pub display: Display<WindowSurface>,
}

pub trait AppBehaviour {
    /// Processes winit events.
    ///
    /// Returns `true` if the app should continue running, `false` otherwise.
    fn process_events(&mut self, event: Event<()>) -> bool;

    /// Updates the app state.
    fn update(&mut self, delta_time: Duration);

    /// Renders the app.
    fn render(&mut self, frame: &mut glium::Frame);
}

pub struct App {
    pub event_loop: EventLoop<()>,
    pub window: Rc<Window>,
    pub should_close: bool,

    last_frame_time: Instant,
    delta_time: Duration,
}

impl App {
    /// Creates a new winit window, and initializes OpenGL.
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        let event_loop = EventLoop::new().expect("to create event loop");
        let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
            .with_title(title)
            .with_inner_size(width, height)
            .build(&event_loop);

        Self {
            event_loop,
            window: Rc::new(Window {
                winit: window,
                display,
            }),
            should_close: false,

            last_frame_time: Instant::now(),
            delta_time: Duration::ZERO,
        }
    }

    /// Run the given app.
    pub fn run(&mut self, mut app: impl AppBehaviour) {
        while !self.should_close {
            let current_time = Instant::now();
            self.delta_time = current_time.duration_since(self.last_frame_time);
            self.last_frame_time = current_time;

            self.event_loop
                .pump_events(Some(Duration::ZERO), |event, _| {
                    match event {
                        Event::WindowEvent {
                            event: WindowEvent::CloseRequested,
                            ..
                        } => {
                            self.should_close = true;
                        }
                        Event::WindowEvent {
                            event: WindowEvent::Resized(new_size),
                            ..
                        } => self.window.display.resize(new_size.into()),
                        _ => {}
                    };

                    if !self.should_close {
                        self.should_close = !app.process_events(event);
                    }
                });
            if self.should_close {
                return;
            }

            app.update(self.delta_time);

            let mut frame = self.window.display.draw();
            frame.clear_color_srgb_and_depth((1.0, 0.0, 1.0, 1.0), 1.0);

            app.render(&mut frame);

            frame.finish().expect("to finish drawing frame");
        }
    }
}
