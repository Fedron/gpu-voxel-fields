use glam::Vec3;
use num_enum::{FromPrimitive, IntoPrimitive};
use strum::{Display, EnumIter};
use vulkano::{buffer::Subbuffer, memory::allocator::DeviceLayout, sync::GpuFuture};

use crate::world::chunk::Chunk;

pub mod brute_force;
pub mod hybrid;

#[repr(u32)]
#[derive(Debug, Clone, Copy, IntoPrimitive, FromPrimitive, PartialEq, EnumIter, Display)]
pub enum Algorithm {
    BruteForce,
    FastIterative,
    JumpFlooding,
    #[num_enum(default)]
    Hybrid,
}

pub trait DistanceFieldPipeline {
    fn compute(
        &self,
        distance_field: Subbuffer<[u8]>,
        chunk: &Chunk,
        chunk_pos: Vec3,
    ) -> Box<dyn GpuFuture>;
    fn execution_time(&self) -> f32;

    fn layout(&self) -> DeviceLayout;
}
