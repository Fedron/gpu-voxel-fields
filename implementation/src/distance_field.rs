use std::sync::Arc;

use fim::FIMDistanceFieldPipeline;
use jump_flooding::JFADistanceFieldPipeline;
use vulkano::{
    buffer::Subbuffer, command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator, device::Queue,
    memory::allocator::StandardMemoryAllocator,
};

use crate::world::chunk::Chunk;

pub mod fim;
pub mod jump_flooding;

pub struct DistanceFieldPipeline {
    jfa_pipeline: JFADistanceFieldPipeline,
    fim_pipeline: FIMDistanceFieldPipeline,
}

impl DistanceFieldPipeline {
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        chunk_size: u32,
    ) -> Self {
        let jfa_pipeline = JFADistanceFieldPipeline::new(
            queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
            chunk_size,
        );
        let fim_pipeline = FIMDistanceFieldPipeline::new(
            queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
        );

        Self {
            jfa_pipeline,
            fim_pipeline,
        }
    }

    /// Computes an approximation of the distance field.
    ///
    /// Quick but can result in artifacts when rendered.
    ///
    /// Returns the execution time of the compute shader in milliseconds.
    pub fn compute_coarse(&self, distance_field: Subbuffer<[u8]>, chunk: &Chunk) -> f32 {
        self.jfa_pipeline.compute(distance_field, chunk);
        self.jfa_pipeline.execution_time()
    }

    /// Computes an exact distance field.
    ///
    /// Assumes the distance field has been initialized in some way i.e. with a coarse computation first.
    ///
    /// Returns the execution time of the compute shader in milliseconds.
    pub fn compute_fine(&self, distance_field: Subbuffer<[u8]>, chunk: &Chunk) -> f32 {
        self.fim_pipeline.compute(distance_field, chunk)
    }
}
