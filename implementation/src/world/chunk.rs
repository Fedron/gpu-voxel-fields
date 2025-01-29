use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::utils::position_to_index;

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

    pub fn get(&self, position: glam::UVec3) -> Option<Voxel> {
        self.voxels
            .read()
            .unwrap()
            .get(position_to_index(position, self.size))
            .and_then(|&v| Some(Voxel::from(v)))
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
}
