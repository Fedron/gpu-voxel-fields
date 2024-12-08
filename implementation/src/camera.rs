use crate::input::InputState;

#[derive(Clone, Copy)]
pub struct Camera {
    pub position: glam::Vec3,
    pitch: f32,
    yaw: f32,

    fov: f32,
    window_size: (u32, u32),

    rotate_horizontal: f32,
    rotate_vertical: f32,
}

impl Camera {
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

            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
        }
    }

    pub fn front(&self) -> glam::Vec3 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        glam::Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize()
    }

    pub fn view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.position, self.position + self.front(), glam::Vec3::Y)
    }

    pub fn projection_matrix(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(
            self.fov,
            self.window_size.0 as f32 / self.window_size.1 as f32,
            0.1,
            1000.0,
        )
    }

    pub fn resize(&mut self, new_size: (u32, u32)) {
        self.window_size = new_size;
    }

    pub fn on_mouse_motion(&mut self, delta: (f64, f64)) {
        self.rotate_horizontal = delta.0 as f32;
        self.rotate_vertical = -delta.1 as f32;
    }

    pub fn handle_input(&mut self, input: &InputState, dt: f32) {
        let front = glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();
        let right = front.cross(glam::Vec3::Y).normalize();

        let move_speed = 10.0 * dt;
        let rotate_speed = 1.0 * dt;

        if input.move_forward {
            self.position += front * move_speed;
        }

        if input.move_backward {
            self.position -= front * move_speed;
        }

        if input.move_right {
            self.position += right * move_speed;
        }

        if input.move_left {
            self.position -= right * move_speed;
        }

        if input.move_up {
            self.position += glam::Vec3::Y * move_speed;
        }

        if input.move_down {
            self.position -= glam::Vec3::Y * move_speed;
        }

        self.yaw += self.rotate_horizontal * rotate_speed;
        self.pitch += self.rotate_vertical * rotate_speed;

        self.pitch = self.pitch.clamp(-89.0, 89.0);

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
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
