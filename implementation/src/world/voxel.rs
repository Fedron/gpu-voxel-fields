use num_enum::{FromPrimitive, IntoPrimitive};
use strum::{Display, EnumIter};

#[repr(u32)]
#[derive(Debug, Clone, Copy, IntoPrimitive, FromPrimitive, PartialEq, EnumIter, Display)]
pub enum Voxel {
    #[num_enum(default)]
    Air = 0,
    Black,
    White,
    Grey,
    Red,
    Pink,
    Purple,
    Blue,
    Cyan,
    Turquoise,
    Green,
    Yellow,
}
