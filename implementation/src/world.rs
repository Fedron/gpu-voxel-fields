use std::sync::Arc;

use chunk::Chunk;
use voxel::Voxel;
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::{ray::Ray, utils::position_to_index};

pub mod chunk;
pub mod voxel;

pub struct World {
    pub update_count: usize,

    chunk_size: glam::UVec3,
    pub num_chunks: glam::UVec3,
    pub chunks: Vec<Chunk>,
}

impl World {
    pub fn new(
        chunk_size: glam::UVec3,
        num_chunks: glam::UVec3,
        memory_allocator: Arc<StandardMemoryAllocator>,
    ) -> Self {
        let mut chunks = Vec::with_capacity(num_chunks.element_product() as usize);
        for _ in 0..num_chunks.element_product() {
            chunks.push(Chunk::new(chunk_size, memory_allocator.clone()));
        }

        Self {
            update_count: 0,

            chunk_size,
            num_chunks,
            chunks,
        }
    }

    /// The size of the world in voxels.
    pub fn size(&self) -> glam::UVec3 {
        self.chunk_size.saturating_mul(self.num_chunks)
    }

    pub fn get(&self, position: glam::UVec3) -> Option<Voxel> {
        let chunk_pos = position.saturating_div(self.chunk_size);
        match self
            .chunks
            .get(position_to_index(chunk_pos, self.num_chunks))
        {
            Some(chunk) => chunk.get(position % self.chunk_size),
            None => None,
        }
    }

    pub fn set(&mut self, position: glam::UVec3, voxel: Voxel) {
        let chunk_pos = position.saturating_div(self.chunk_size);
        if let Some(chunk) = self
            .chunks
            .get_mut(position_to_index(chunk_pos, self.num_chunks))
        {
            if chunk.set(position % self.chunk_size, voxel) {
                self.update_count += 1;
            }
        }
    }

    /// Casts a ray through the world and returns the first solid voxel that is intersected.
    pub fn is_voxel_hit(&self, ray: Ray) -> Hit {
        let hit = ray.intersect_aabb(glam::Vec3::ZERO, self.size().as_vec3());
        if !hit.does_intersect {
            return Hit {
                does_intersect: false,
                voxel_position: None,
                face_normal: None,
            };
        }

        let start_pos = ray.position + hit.near * ray.direction;
        let mut voxel_position = start_pos.max(glam::Vec3::ZERO);
        let mut face_normal = glam::Vec3::ZERO;

        let ray_step = ray.direction.signum();
        let delta_dist = (1.0 / ray.direction).abs();

        let mut side_dist = glam::Vec3::new(
            if ray.direction.x > 0.0 {
                voxel_position.x.floor() + 1.0 - start_pos.x
            } else {
                start_pos.x - voxel_position.x.floor()
            },
            if ray.direction.y > 0.0 {
                voxel_position.y.floor() + 1.0 - start_pos.y
            } else {
                start_pos.y - voxel_position.y.floor()
            },
            if ray.direction.z > 0.0 {
                voxel_position.z.floor() + 1.0 - start_pos.z
            } else {
                start_pos.z - voxel_position.z.floor()
            },
        ) * delta_dist;

        while self.is_in_bounds(voxel_position) {
            match self.get(voxel_position.as_uvec3()) {
                Some(voxel) => {
                    if voxel != Voxel::Air {
                        return Hit {
                            does_intersect: true,
                            voxel_position: Some(voxel_position.as_uvec3()),
                            face_normal: Some(face_normal.as_ivec3()),
                        };
                    }
                }
                None => {
                    return Hit {
                        does_intersect: false,
                        voxel_position: None,
                        face_normal: None,
                    };
                }
            }

            if side_dist.x < side_dist.y && side_dist.x < side_dist.z {
                side_dist.x += delta_dist.x;
                voxel_position.x += ray_step.x;
                face_normal = glam::vec3(-ray_step.x, 0.0, 0.0);
            } else if side_dist.y < side_dist.z {
                side_dist.y += delta_dist.y;
                voxel_position.y += ray_step.y;
                face_normal = glam::vec3(0.0, -ray_step.y, 0.0);
            } else {
                side_dist.z += delta_dist.z;
                voxel_position.z += ray_step.z;
                face_normal = glam::vec3(0.0, 0.0, -ray_step.z);
            }
        }

        Hit {
            does_intersect: false,
            voxel_position: None,
            face_normal: None,
        }
    }

    fn is_in_bounds(&self, position: glam::Vec3) -> bool {
        position.x >= 0.0
            && position.x < self.size().x as f32
            && position.y >= 0.0
            && position.y < self.size().y as f32
            && position.z >= 0.0
            && position.z < self.size().z as f32
    }

    pub fn is_in_bounds_ivec3(&self, position: glam::IVec3) -> bool {
        position.x >= 0
            && position.x < self.size().x as i32
            && position.y >= 0
            && position.y < self.size().y as i32
            && position.z >= 0
            && position.z < self.size().z as i32
    }
}

impl Into<crate::ray_marcher_pipeline::cs::World> for &World {
    fn into(self) -> crate::ray_marcher_pipeline::cs::World {
        crate::ray_marcher_pipeline::cs::World {
            size: Into::<[u32; 3]>::into(self.size()).into(),
            chunk_size: Into::<[u32; 3]>::into(self.chunk_size).into(),
            num_chunks: self.num_chunks.into(),
        }
    }
}

#[derive(Debug)]
pub struct Hit {
    pub does_intersect: bool,
    pub voxel_position: Option<glam::UVec3>,
    pub face_normal: Option<glam::IVec3>,
}
