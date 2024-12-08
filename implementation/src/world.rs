use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::ray::Ray;

#[derive(Clone, Copy)]
pub enum Voxel {
    Air = 0,
    Stone = 1,
}

#[derive(Clone)]
pub struct World<const X: usize, const Y: usize, const Z: usize>
where
    [(); X * Y * Z]:,
{
    pub voxels: Subbuffer<[u32]>,
    pub is_dirty: bool,
}

impl<const X: usize, const Y: usize, const Z: usize> World<X, Y, Z>
where
    [(); X * Y * Z]:,
{
    /// Creates a new host-visible buffer that will store the world's voxels.
    pub fn new(memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
        let voxels = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0; X * Y * Z],
        )
        .unwrap();

        Self {
            voxels,
            is_dirty: false,
        }
    }

    pub fn size(&self) -> [u32; 3] {
        [X as u32, Y as u32, Z as u32]
    }

    /// Updates the voxel at the given position.
    pub fn set(&mut self, position: glam::UVec3, voxel: Voxel) {
        let mut voxels = self.voxels.write().unwrap();
        if let Some(v) = voxels.get_mut(self.position_to_index(position)) {
            *v = voxel as u32;
            self.is_dirty = true;
        }
    }

    pub fn is_voxel_hit(&self, ray: Ray) -> Hit {
        let hit = ray.intersect_aabb(glam::Vec3::ZERO, glam::vec3(X as f32, Y as f32, Z as f32));
        if !hit.does_intersect {
            return Hit {
                does_intersect: false,
                voxel_position: None,
                face_normal: None,
            };
        }

        let voxels = self.voxels.read().unwrap();

        let mut voxel_position = ray.position.clamp(
            glam::Vec3::ZERO,
            glam::vec3(X as f32 - 1.0, Y as f32 - 1.0, Z as f32 - 1.0),
        );
        let mut face_normal = glam::Vec3::ZERO;

        let delta_dist = ray.inverse_direction.abs();
        let ray_step = ray.direction.signum();
        let mut side_dist = (ray.direction.signum() * (voxel_position - ray.position)
            + (ray.direction.signum() * 0.5)
            + 0.5)
            * ray.inverse_direction.abs();

        while self.is_in_bounds(voxel_position.as_uvec3()) {
            match voxels.get(self.position_to_index(voxel_position.as_uvec3())) {
                Some(&voxel) => {
                    if voxel != Voxel::Air as u32 {
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
                    }
                }
            }

            if side_dist.x < side_dist.y {
                if side_dist.x < side_dist.z {
                    side_dist.x += delta_dist.x;
                    voxel_position.x += ray_step.x;
                    face_normal = glam::vec3(-ray_step.x, 0.0, 0.0);
                } else {
                    side_dist.z += delta_dist.z;
                    voxel_position.z += ray_step.z;
                    face_normal = glam::vec3(0.0, 0.0, -ray_step.z);
                }
            } else {
                if side_dist.y < side_dist.z {
                    side_dist.y += delta_dist.y;
                    voxel_position.y += ray_step.y;
                    face_normal = glam::vec3(0.0, -ray_step.y, 0.0);
                } else {
                    side_dist.z += delta_dist.z;
                    voxel_position.z += ray_step.z;
                    face_normal = glam::vec3(0.0, 0.0, -ray_step.z);
                }
            }
        }

        Hit {
            does_intersect: false,
            voxel_position: None,
            face_normal: None,
        }
    }

    fn position_to_index(&self, position: glam::UVec3) -> usize {
        position.x as usize + position.y as usize * X + position.z as usize * X * Y
    }

    fn is_in_bounds(&self, position: glam::UVec3) -> bool {
        (position.x as usize) < X && (position.y as usize) < Y && (position.z as usize) < Z
    }
}

impl<const X: usize, const Y: usize, const Z: usize> Into<crate::ray_marcher_pipeline::cs::World>
    for World<X, Y, Z>
where
    [(); X * Y * Z]:,
{
    fn into(self) -> crate::ray_marcher_pipeline::cs::World {
        crate::ray_marcher_pipeline::cs::World {
            min: [0.0; 3].into(),
            max: [X as f32, Y as f32, Z as f32].into(),
            size: [X as i32, Y as i32, Z as i32],
        }
    }
}

impl<const X: usize, const Y: usize, const Z: usize> Into<crate::ray_marcher_pipeline::cs::World>
    for &World<X, Y, Z>
where
    [(); X * Y * Z]:,
{
    fn into(self) -> crate::ray_marcher_pipeline::cs::World {
        crate::ray_marcher_pipeline::cs::World {
            min: [0.0; 3].into(),
            max: [X as f32, Y as f32, Z as f32].into(),
            size: [X as i32, Y as i32, Z as i32],
        }
    }
}

pub struct Hit {
    pub does_intersect: bool,
    pub voxel_position: Option<glam::UVec3>,
    pub face_normal: Option<glam::IVec3>,
}
