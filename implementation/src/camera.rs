use winit::{event::ElementState, keyboard::KeyCode};

pub struct Camera {
    pub position: glam::Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    pub fn new(position: glam::Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
        }
    }

    pub fn view_matrix(&self) -> glam::Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let front =
            glam::Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();

        glam::Mat4::look_at_rh(self.position, self.position + front, glam::Vec3::Y)
    }
}

pub struct Projection {
    aspect: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl Projection {
    pub fn new(aspect: f32, fov: f32, near: f32, far: f32) -> Self {
        Self {
            aspect,
            fov,
            near,
            far,
        }
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.aspect = width / height;
    }

    pub fn matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fov.to_radians(), self.aspect, self.near, self.far)
    }
}

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

    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        self.rotate_horizontal = mouse_dx;
        self.rotate_vertical = -mouse_dy;
    }

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
