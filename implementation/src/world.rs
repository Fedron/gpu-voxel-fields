use std::{collections::HashMap, sync::Arc};

use num_enum::{FromPrimitive, IntoPrimitive};
use rand::{seq::SliceRandom, thread_rng};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::ray::Ray;

#[repr(u32)]
#[derive(Clone, Copy, IntoPrimitive, FromPrimitive, PartialEq)]
pub enum Voxel {
    #[num_enum(default)]
    Air = 0,
    Stone = 1,
    Sand = 2,
    Water = 3,
    SandGenerator = 4,
    WaterGenerator = 5,
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

    /// Runs a step of the "Falling Sand" simulation.
    pub fn update(&mut self) {
        let mut voxels = self.voxels.write().unwrap();
        let mut staged_updates = StagingList::new();

        for x in 0..X {
            for y in 0..Y {
                for z in 0..Z {
                    let voxel = Voxel::from(
                        voxels[self
                            .position_to_index(glam::uvec3(x as u32, y as u32, z as u32))
                            .unwrap()],
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
                                if let Some(index) = self.position_to_index(neighbour) {
                                    let neighbour_voxel = Voxel::from(voxels[index]);
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
                                if let Some(index) = self.position_to_index(neighbour) {
                                    if Voxel::from(voxels[index]) == Voxel::Air {
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
                            if let Some(index) = self.position_to_index(neighbour) {
                                if Voxel::from(voxels[index]) == Voxel::Air {
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
                            if let Some(index) = self.position_to_index(neighbour) {
                                if Voxel::from(voxels[index]) == Voxel::Air {
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

        if !staged_updates.is_empty() {
            self.is_dirty = true;
        }

        for update in staged_updates.values() {
            voxels[self.position_to_index(update.from).unwrap()] = update.replace_old_with.into();
            voxels[self.position_to_index(update.to).unwrap()] = update.new_state.into();
        }
    }

    /// Updates the voxel at the given position.
    pub fn set(&mut self, position: glam::UVec3, voxel: Voxel) {
        let mut voxels = self.voxels.write().unwrap();
        if let Some(v) = voxels.get_mut(self.position_to_index(position).unwrap()) {
            *v = voxel.into();
            self.is_dirty = true;
        }
    }

    /// Casts a ray through the world and returns the first solid voxel that is intersected.
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
            match voxels.get(self.position_to_index(voxel_position.as_uvec3()).unwrap()) {
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

    fn position_to_index(&self, position: glam::UVec3) -> Option<usize> {
        if position.x as usize >= X || position.y as usize >= Y || position.z as usize >= Z {
            return None;
        }

        Some(position.x as usize + position.y as usize * X + position.z as usize * X * Y)
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

struct StagedUpdate {
    from: glam::UVec3,
    to: glam::UVec3,
    replace_old_with: Voxel,
    new_state: Voxel,
}

type StagingList = HashMap<glam::UVec3, StagedUpdate>;
