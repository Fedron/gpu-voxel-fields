#[derive(Debug, Clone, Copy)]
/// Represents the hit information of a ray.
pub struct Hit {
    pub does_intersect: bool,
    pub near: f32,
    pub _far: f32,
}

#[derive(Debug, Clone, Copy)]
/// A single ray that can be operated on in 3D.
pub struct Ray {
    pub position: glam::Vec3,
    pub direction: glam::Vec3,
    pub inverse_direction: glam::Vec3,
}

impl Ray {
    /// Creates a new ray with a `position` and `direction`.
    pub fn new(position: glam::Vec3, direction: glam::Vec3) -> Self {
        let inverse_direction = glam::Vec3::ONE.div_euclid(direction);
        Self {
            position,
            direction,
            inverse_direction,
        }
    }

    /// Moves the `position` in the `direction` of the ray.
    pub fn advance(&mut self, amount: f32) {
        self.position += self.direction * amount;
    }

    /// Tests whether the ray will intersect with an AABB.
    pub fn intersect_aabb(&self, box_min: glam::Vec3, box_max: glam::Vec3) -> Hit {
        let t_min = (box_min - self.position) * self.inverse_direction;
        let t_max = (box_max - self.position) * self.inverse_direction;

        let t1 = t_min.min(t_max);
        let t2 = t_min.max(t_max);

        let near = t1.x.max(t1.y.max(t1.z));
        let far = t2.x.min(t2.y.min(t2.z));

        Hit {
            does_intersect: near <= far,
            near,
            _far: far,
        }
    }
}
