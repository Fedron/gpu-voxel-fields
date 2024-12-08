use glam::f32::Vec2;
use vulkano_util::window::WindowDescriptor;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    keyboard::{Key, NamedKey},
};

pub struct InputState {
    pub window_size: [f32; 2],
    pub should_quit: bool,
    pub mouse_pos: Vec2,
    last_mouse_pos: Vec2,

    pub move_forward: bool,
    pub move_right: bool,
    pub move_backward: bool,
    pub move_left: bool,
    pub move_up: bool,
    pub move_down: bool,
}

impl InputState {
    pub fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
            should_quit: false,
            mouse_pos: Vec2::ZERO,
            last_mouse_pos: Vec2::ZERO,
            move_forward: false,
            move_right: false,
            move_backward: false,
            move_left: false,
            move_up: false,
            move_down: false,
        }
    }

    pub fn mouse_diff(&self) -> Vec2 {
        self.mouse_pos - self.last_mouse_pos
    }

    /// Resets values that should be reset. All incremental mappings and toggles should be reset.
    pub fn reset(&mut self) {
        *self = InputState { ..*self }
    }

    pub fn handle_input(&mut self, window_size: PhysicalSize<u32>, event: &WindowEvent) {
        self.window_size = window_size.into();

        match event {
            WindowEvent::KeyboardInput { event, .. } => self.on_keyboard_event(event),
            WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
            _ => {}
        }
    }

    /// Matches keyboard events to our defined inputs.
    fn on_keyboard_event(&mut self, event: &KeyEvent) {
        match event.logical_key.as_ref() {
            Key::Named(NamedKey::Escape) => self.should_quit = event.state.is_pressed(),
            Key::Character("w") => self.move_forward = event.state.is_pressed(),
            Key::Character("a") => self.move_left = event.state.is_pressed(),
            Key::Character("s") => self.move_backward = event.state.is_pressed(),
            Key::Character("d") => self.move_right = event.state.is_pressed(),
            Key::Named(NamedKey::Space) => self.move_up = event.state.is_pressed(),
            Key::Named(NamedKey::Control) => self.move_down = event.state.is_pressed(),
            _ => {}
        }
    }

    /// Update mouse position
    fn on_cursor_moved_event(&mut self, pos: &PhysicalPosition<f64>) {
        self.last_mouse_pos = self.mouse_pos;
        self.mouse_pos = Vec2::new(pos.x as f32, pos.y as f32);
    }
}
