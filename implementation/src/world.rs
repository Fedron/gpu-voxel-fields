use std::{collections::HashMap, sync::Arc};

use chunk::Chunk;
use rand::{seq::SliceRandom, thread_rng};
use voxel::Voxel;
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::utils::{index_to_position, position_to_index};

pub mod chunk;
pub mod voxel;

pub struct World {
    pub update_count: usize,

    chunk_size: glam::UVec3,
    num_chunks: glam::UVec3,
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

    pub fn update(&mut self) {
        let mut staged_updates = StagingList::new();
        for (index, chunk) in self.chunks.iter().enumerate() {
            let chunk_pos = index_to_position(index as u32, self.num_chunks);
            for x in 0..chunk.size.x {
                for y in 0..chunk.size.y {
                    for z in 0..chunk.size.z {
                        let voxel_pos = glam::uvec3(x, y, z);
                        let global_voxel_pos = (chunk_pos * self.chunk_size) + voxel_pos;

                        let voxel = chunk.get(voxel_pos).unwrap();

                        match voxel {
                            Voxel::Sand | Voxel::Water => {
                                let mut neighbour_offsets = voxel.move_directions().unwrap();
                                neighbour_offsets.shuffle(&mut thread_rng());

                                'ncheck: for offset in neighbour_offsets {
                                    let neighbour_voxel_pos = voxel_pos.as_ivec3() + offset;
                                    let global_neighbour_voxel_pos = (chunk_pos * self.chunk_size)
                                        .as_ivec3()
                                        + neighbour_voxel_pos;

                                    let neighbour_voxel = {
                                        if global_neighbour_voxel_pos.x < 0
                                            || global_neighbour_voxel_pos.y < 0
                                            || global_neighbour_voxel_pos.z < 0
                                        {
                                            None
                                        } else {
                                            self.get(global_neighbour_voxel_pos.as_uvec3())
                                        }
                                    };

                                    if let Some(neighbour_voxel) = neighbour_voxel {
                                        if voxel.can_replace(neighbour_voxel) {
                                            match staged_updates.try_insert(
                                                global_neighbour_voxel_pos.as_uvec3(),
                                                StagedUpdate {
                                                    from: global_voxel_pos,
                                                    to: global_neighbour_voxel_pos.as_uvec3(),
                                                    replace_old_with: neighbour_voxel,
                                                    new_state: voxel,
                                                },
                                            ) {
                                                Ok(_) => break 'ncheck,
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                            Voxel::SandGenerator | Voxel::WaterGenerator => {
                                let neighbour_voxel_pos =
                                    voxel_pos.as_ivec3() + glam::ivec3(0, -1, 0);
                                let global_neighbour_voxel_pos = ((chunk_pos * self.chunk_size)
                                    .as_ivec3()
                                    + neighbour_voxel_pos)
                                    .as_uvec3();

                                let neighbour_voxel = {
                                    if neighbour_voxel_pos.x < 0
                                        || neighbour_voxel_pos.y < 0
                                        || neighbour_voxel_pos.z < 0
                                    {
                                        self.get(global_neighbour_voxel_pos)
                                    } else {
                                        match chunk.get(neighbour_voxel_pos.as_uvec3()) {
                                            Some(neighbour_voxel) => Some(neighbour_voxel),
                                            None => self.get(global_neighbour_voxel_pos),
                                        }
                                    }
                                };

                                if let Some(neighbour_voxel) = neighbour_voxel {
                                    if Voxel::from(neighbour_voxel) == Voxel::Air {
                                        let _ = staged_updates.try_insert(
                                            global_neighbour_voxel_pos,
                                            StagedUpdate {
                                                from: global_voxel_pos,
                                                to: global_neighbour_voxel_pos,
                                                replace_old_with: voxel,
                                                new_state: voxel.generates().unwrap(),
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
        }

        for update in staged_updates.values() {
            self.set(update.from, update.replace_old_with);
            self.set(update.to, update.new_state);
        }
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

pub struct Hit {
    pub does_intersect: bool,
    pub voxel_position: Option<glam::UVec3>,
    pub face_normal: Option<glam::IVec3>,
}

#[derive(Debug, Clone, Copy)]
struct StagedUpdate {
    from: glam::UVec3,
    to: glam::UVec3,
    replace_old_with: Voxel,
    new_state: Voxel,
}

type StagingList = HashMap<glam::UVec3, StagedUpdate>;
