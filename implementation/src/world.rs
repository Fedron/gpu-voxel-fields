use glium::implement_uniform_block;

#[derive(Clone, Copy)]
pub enum Voxel {
    Air = 0,
    Stone = 1,
}

const WORLD_SIZE: usize = 8 * 8 * 8;

#[derive(Clone, Copy)]
pub struct VoxelGrid {
    voxels: [u32; WORLD_SIZE],
}
implement_uniform_block!(VoxelGrid, voxels);

impl VoxelGrid {
    pub fn set(&mut self, position: glam::UVec3, voxel: Voxel) {
        if position.x >= 8 || position.y >= 8 || position.z >= 8 {
            return;
        }

        let index = self.position_to_index(position);
        self.voxels[index] = voxel as u32;
    }

    fn position_to_index(&self, position: glam::UVec3) -> usize {
        (position.x + (position.y * 8) + (position.z * 8 * 8)) as usize
    }
}

#[derive(Clone, Copy)]
pub struct DistanceField {
    pub distance_field: [u32; WORLD_SIZE],
}
implement_uniform_block!(DistanceField, distance_field);
