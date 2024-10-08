#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub min: glam::UVec3,
    pub max: glam::UVec3,
}

impl Bounds {
    pub fn contains(&self, position: glam::UVec3) -> bool {
        position.x >= self.min.x
            && position.y >= self.min.y
            && position.z >= self.min.z
            && position.x <= self.max.x
            && position.y <= self.max.y
            && position.z <= self.max.z
    }
}
