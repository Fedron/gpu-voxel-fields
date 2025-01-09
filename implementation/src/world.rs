use std::sync::Arc;

use chunk::Chunk;
use voxel::Voxel;
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::utils::position_to_index;

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
        for chunk in self.chunks.iter_mut() {
            if chunk.update() {
                self.update_count += 1;
            }
        }
    }
}

impl Into<crate::ray_marcher_pipeline::cs::World> for &World {
    fn into(self) -> crate::ray_marcher_pipeline::cs::World {
        crate::ray_marcher_pipeline::cs::World {
            min: [0.0; 3].into(),
            max: Into::<[f32; 3]>::into(self.size().as_vec3()).into(),
            size: Into::<[i32; 3]>::into(self.size().as_ivec3()).into(),
            chunk_size: Into::<[u32; 3]>::into(self.chunk_size).into(),
            num_chunks: self.num_chunks.into(),
        }
    }
}
