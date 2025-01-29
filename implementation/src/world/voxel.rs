use num_enum::{FromPrimitive, IntoPrimitive};

#[repr(u32)]
#[derive(Debug, Clone, Copy, IntoPrimitive, FromPrimitive, PartialEq)]
pub enum Voxel {
    #[num_enum(default)]
    Air = 0,
    Stone = 1,
    Sand = 2,
    Water = 3,
}
