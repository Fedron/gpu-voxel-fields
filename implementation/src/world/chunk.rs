use std::{collections::HashMap, sync::Arc};

use rand::{seq::SliceRandom, thread_rng};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::{ray::Ray, utils::position_to_index};

use super::voxel::Voxel;

#[derive(Clone)]
pub struct Chunk {
    pub size: glam::UVec3,
    pub voxels: Subbuffer<[u32]>,
    pub is_dirty: bool,
}

impl Chunk {
    /// Creates a new host-visible buffer that will store the chunk's voxels.
    pub fn new(size: glam::UVec3, memory_allocator: Arc<StandardMemoryAllocator>) -> Self {
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
            vec![0; size.element_product() as usize],
        )
        .unwrap();

        Self {
            size,
            voxels,
            is_dirty: true,
        }
    }

    /// Runs a step of the "Falling Sand" simulation.
    ///
    /// Returns whether any voxel was updated.
    pub fn update(&mut self) -> bool {
        let mut voxels = self.voxels.write().unwrap();
        let mut staged_updates = StagingList::new();

        for x in 0..self.size.x {
            for y in 0..self.size.y {
                for z in 0..self.size.z {
                    let voxel = Voxel::from(
                        voxels[position_to_index(
                            glam::uvec3(x as u32, y as u32, z as u32),
                            self.size,
                        )],
                    );

                    match voxel {
                        Voxel::Sand => {
                            if y == 0 {
                                continue;
                            }

                            let mut neighbour_offsets = vec![
                                (-1, -1, -1),
                                (-1, -1, 0),
                                (-1, -1, 1),
                                (0, -1, -1),
                                (0, -1, 0),
                                (0, -1, 1),
                                (1, -1, -1),
                                (1, -1, 0),
                                (1, -1, 1),
                            ];
                            neighbour_offsets.shuffle(&mut thread_rng());

                            'ncheck: for &(nx, ny, nz) in &neighbour_offsets {
                                let neighbour = glam::uvec3(
                                    x.saturating_add_signed(nx) as u32,
                                    y.saturating_add_signed(ny) as u32,
                                    z.saturating_add_signed(nz) as u32,
                                );
                                if let Some(&voxel) =
                                    voxels.get(position_to_index(neighbour, self.size))
                                {
                                    let neighbour_voxel = Voxel::from(voxel);
                                    if neighbour_voxel == Voxel::Air
                                        || neighbour_voxel == Voxel::Water
                                    {
                                        match staged_updates.try_insert(
                                            neighbour,
                                            StagedUpdate {
                                                from: glam::uvec3(x as u32, y as u32, z as u32),
                                                to: neighbour,
                                                replace_old_with: neighbour_voxel,
                                                new_state: Voxel::Sand,
                                            },
                                        ) {
                                            Ok(_) => break 'ncheck,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        Voxel::Water => {
                            if y == 0 {
                                continue;
                            }

                            let mut neighbour_offsets = vec![
                                (-1, 0, -1),
                                (-1, 0, 0),
                                (-1, 0, 1),
                                (0, 0, -1),
                                (0, 0, 1),
                                (1, 0, -1),
                                (1, 0, 0),
                                (1, 0, 1),
                                (-1, -1, -1),
                                (-1, -1, 0),
                                (-1, -1, 1),
                                (0, -1, -1),
                                (0, -1, 0),
                                (0, -1, 1),
                                (1, -1, -1),
                                (1, -1, 0),
                                (1, -1, 1),
                            ];
                            neighbour_offsets.shuffle(&mut thread_rng());

                            'ncheck: for &(nx, ny, nz) in &neighbour_offsets {
                                let neighbour = glam::uvec3(
                                    x.saturating_add_signed(nx) as u32,
                                    y.saturating_add_signed(ny) as u32,
                                    z.saturating_add_signed(nz) as u32,
                                );
                                if let Some(&voxel) =
                                    voxels.get(position_to_index(neighbour, self.size))
                                {
                                    if Voxel::from(voxel) == Voxel::Air {
                                        match staged_updates.try_insert(
                                            neighbour,
                                            StagedUpdate {
                                                from: glam::uvec3(x as u32, y as u32, z as u32),
                                                to: neighbour,
                                                replace_old_with: Voxel::Air,
                                                new_state: Voxel::Water,
                                            },
                                        ) {
                                            Ok(_) => break 'ncheck,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                        Voxel::SandGenerator => {
                            if y == 0 {
                                continue;
                            }

                            let neighbour =
                                glam::uvec3(x as u32, y.saturating_sub(1) as u32, z as u32);
                            if let Some(&voxel) =
                                voxels.get(position_to_index(neighbour, self.size))
                            {
                                if Voxel::from(voxel) == Voxel::Air {
                                    let _ = staged_updates.try_insert(
                                        neighbour,
                                        StagedUpdate {
                                            from: glam::uvec3(x as u32, y as u32, z as u32),
                                            to: neighbour,
                                            replace_old_with: Voxel::SandGenerator,
                                            new_state: Voxel::Sand,
                                        },
                                    );
                                }
                            }
                        }
                        Voxel::WaterGenerator => {
                            if y == 0 {
                                continue;
                            }

                            let neighbour =
                                glam::uvec3(x as u32, y.saturating_sub(1) as u32, z as u32);

                            if let Some(&voxel) =
                                voxels.get(position_to_index(neighbour, self.size))
                            {
                                if Voxel::from(voxel) == Voxel::Air {
                                    let _ = staged_updates.try_insert(
                                        neighbour,
                                        StagedUpdate {
                                            from: glam::uvec3(x as u32, y as u32, z as u32),
                                            to: neighbour,
                                            replace_old_with: Voxel::WaterGenerator,
                                            new_state: Voxel::Water,
                                        },
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let updated = if !staged_updates.is_empty() {
            self.is_dirty = true;
            true
        } else {
            false
        };

        for update in staged_updates.values() {
            voxels[position_to_index(update.from, self.size)] = update.replace_old_with.into();
            voxels[position_to_index(update.to, self.size)] = update.new_state.into();
        }

        updated
    }

    /// Updates the voxel at the given position.
    ///
    /// Returns whether the `position` was updated.
    pub fn set(&mut self, position: glam::UVec3, voxel: Voxel) -> bool {
        let mut voxels = self.voxels.write().unwrap();
        if let Some(v) = voxels.get_mut(position_to_index(position, self.size)) {
            *v = voxel.into();
            self.is_dirty = true;
            return true;
        }

        false
    }

    /// Casts a ray through the world and returns the first solid voxel that is intersected.
    pub fn is_voxel_hit(&self, ray: Ray) -> Hit {
        let hit = ray.intersect_aabb(glam::Vec3::ZERO, self.size.as_vec3());
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
            self.size.saturating_sub(glam::UVec3::splat(1)).as_vec3(),
        );
        let mut face_normal = glam::Vec3::ZERO;

        let delta_dist = ray.inverse_direction.abs();
        let ray_step = ray.direction.signum();
        let mut side_dist = (ray.direction.signum() * (voxel_position - ray.position)
            + (ray.direction.signum() * 0.5)
            + 0.5)
            * ray.inverse_direction.abs();

        while self.is_in_bounds(voxel_position.as_uvec3()) {
            match voxels.get(position_to_index(voxel_position.as_uvec3(), self.size)) {
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

    fn is_in_bounds(&self, position: glam::UVec3) -> bool {
        position.x < self.size.x && position.y < self.size.y && position.z < self.size.z
    }
}

pub struct Hit {
    pub does_intersect: bool,
    pub voxel_position: Option<glam::UVec3>,
    pub face_normal: Option<glam::IVec3>,
}

struct StagedUpdate {
    from: glam::UVec3,
    to: glam::UVec3,
    replace_old_with: Voxel,
    new_state: Voxel,
}

type StagingList = HashMap<glam::UVec3, StagedUpdate>;
