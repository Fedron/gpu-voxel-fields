use winit::{event::ElementState, keyboard::KeyCode};

/// A three-dimensional perspective camera.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: glam::Vec3,
    pitch: f32,
    yaw: f32,

    fov: f32,
    window_size: (u32, u32),
}

impl Camera {
    /// Creates a new [`Camera`].
    pub fn new(
        position: glam::Vec3,
        pitch: f32,
        yaw: f32,
        fov: f32,
        window_size: (u32, u32),
    ) -> Self {
        Self {
            position,
            pitch,
            yaw,

            fov,
            window_size,
        }
    }

    /// Calculates the front-facing direction.
    pub fn front(&self) -> glam::Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        glam::Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize()
    }

    /// Calculates the view matrix (using a right-handed coordinate system).
    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.position, self.position + self.front(), glam::Vec3::Y)
    }

    /// Calculates the projection matrix (using a right-handed coordinate system).
    pub fn projection_matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(
            self.fov,
            self.window_size.0 as f32 / self.window_size.1 as f32,
            0.1,
            100000.0,
        )
    }

    /// Updates the window size of the camera.
    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.window_size = new_size;
    }
}

impl Into<crate::ray_marcher_pipeline::cs::Camera> for Camera {
    fn into(self) -> crate::ray_marcher_pipeline::cs::Camera {
        crate::ray_marcher_pipeline::cs::Camera {
            position: self.position.to_array().into(),
            inverse_view: self.view_matrix().inverse().to_cols_array_2d(),
            inverse_projection: self.projection_matrix().inverse().to_cols_array_2d(),
        }
    }
}

/// Handles input events to move and update a [`Camera`].
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    current_speed: f32,
    original_speed: f32,
    sensitivity: f32,
}

impl CameraController {
    /// Creates a new controller.
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            current_speed: speed,
            original_speed: speed,
            sensitivity,
        }
    }

    /// Handles keyboard events.
    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) {
        if key == KeyCode::ControlLeft {
            self.current_speed = if state == ElementState::Pressed {
                self.original_speed * 4.0
            } else {
                self.original_speed
            };
        }

        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
            }
            KeyCode::Space => {
                self.amount_up = amount;
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
            }
            _ => (),
        }
    }

    /// Handles mouse movement events.
    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        self.rotate_horizontal = mouse_dx;
        self.rotate_vertical = -mouse_dy;
    }

    /// Updates a [`Camera`], using input information gathered previously using [`Camera::process_keyboard`] and [`Camera::process_mouse`].
    pub fn update_camera(&mut self, camera: &mut Camera, delta_time: f32) {
        let front = glam::Vec3::new(
            camera.yaw.cos() * camera.pitch.cos(),
            camera.pitch.sin(),
            camera.yaw.sin() * camera.pitch.cos(),
        )
        .normalize();
        let right = front.cross(glam::Vec3::Y).normalize();

        let move_speed = self.current_speed * delta_time;
        let rotate_speed = self.sensitivity * delta_time;

        camera.position += front * (self.amount_forward - self.amount_backward) * move_speed;
        camera.position += right * (self.amount_right - self.amount_left) * move_speed;
        camera.position += glam::Vec3::Y * (self.amount_up - self.amount_down) * move_speed;

        camera.yaw += self.rotate_horizontal * rotate_speed;
        camera.pitch += self.rotate_vertical * rotate_speed;

        camera.pitch = camera.pitch.clamp(-89.0, 89.0);

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
    }
}
