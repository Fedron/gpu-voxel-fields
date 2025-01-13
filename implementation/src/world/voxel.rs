use num_enum::{FromPrimitive, IntoPrimitive};

#[repr(u32)]
#[derive(Debug, Clone, Copy, IntoPrimitive, FromPrimitive, PartialEq)]
pub enum Voxel {
    #[num_enum(default)]
    Air = 0,
    Stone = 1,
    Sand = 2,
    Water = 3,
    SandGenerator = 4,
    WaterGenerator = 5,
}

impl Voxel {
    pub fn move_directions(&self) -> Option<Vec<glam::IVec3>> {
        match self {
            Voxel::Sand => Some(vec![
                glam::ivec3(-1, -1, -1),
                glam::ivec3(-1, -1, 0),
                glam::ivec3(-1, -1, 1),
                glam::ivec3(0, -1, -1),
                glam::ivec3(0, -1, 0),
                glam::ivec3(0, -1, 1),
                glam::ivec3(1, -1, -1),
                glam::ivec3(1, -1, 0),
                glam::ivec3(1, -1, 1),
            ]),
            Voxel::Water => Some(vec![
                glam::ivec3(-1, 0, -1),
                glam::ivec3(-1, 0, 0),
                glam::ivec3(-1, 0, 1),
                glam::ivec3(0, 0, -1),
                glam::ivec3(0, 0, 1),
                glam::ivec3(1, 0, -1),
                glam::ivec3(1, 0, 0),
                glam::ivec3(1, 0, 1),
                glam::ivec3(-1, -1, -1),
                glam::ivec3(-1, -1, 0),
                glam::ivec3(-1, -1, 1),
                glam::ivec3(0, -1, -1),
                glam::ivec3(0, -1, 0),
                glam::ivec3(0, -1, 1),
                glam::ivec3(1, -1, -1),
                glam::ivec3(1, -1, 0),
                glam::ivec3(1, -1, 1),
            ]),
            Voxel::SandGenerator => todo!(),
            Voxel::WaterGenerator => todo!(),
            _ => None,
        }
    }

    pub fn can_replace(&self, voxel: Voxel) -> bool {
        match self {
            Voxel::Sand => voxel == Voxel::Air || voxel == Voxel::Water,
            Voxel::Water => voxel == Voxel::Air,
            _ => false,
        }
    }

    pub fn generates(&self) -> Option<Voxel> {
        match self {
            Voxel::SandGenerator => Some(Voxel::Sand),
            Voxel::WaterGenerator => Some(Voxel::Water),
            _ => None,
        }
    }
}
