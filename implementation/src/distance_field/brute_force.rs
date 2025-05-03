use std::sync::Arc;

use glam::Vec3;
use vulkano::{
    buffer::{BufferContents, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    sync::{GpuFuture, PipelineStage},
};

use crate::world::chunk::Chunk;

use super::DistanceFieldPipeline;

pub struct BruteForcePipeline {
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    query_pool: Arc<QueryPool>,

    pipeline: Arc<ComputePipeline>,
    chunk_size: u32,
}

impl BruteForcePipeline {
    pub fn new(
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        chunk_size: u32,
    ) -> Self {
        let pipeline = {
            let device = queue.device();
            let cs = cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let query_pool = QueryPool::new(
            queue.device().clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .unwrap();

        Self {
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            query_pool,

            pipeline,
            chunk_size,
        }
    }
}

impl DistanceFieldPipeline for BruteForcePipeline {
    fn compute(
        &self,
        distance_field: Subbuffer<[u8]>,
        chunk: &Chunk,
        _chunk_pos: Vec3,
    ) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        unsafe {
            builder
                .reset_query_pool(self.query_pool.clone(), 0..2)
                .unwrap()
                .write_timestamp(self.query_pool.clone(), 0, PipelineStage::ComputeShader)
                .unwrap()
        };

        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, chunk.voxels.clone()),
                WriteDescriptorSet::buffer(1, distance_field),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    chunk_size: chunk.size.into(),
                },
            )
            .unwrap();

        unsafe {
            builder
                .dispatch(chunk.size.saturating_div(glam::UVec3::splat(8)).into())
                .unwrap()
                .write_timestamp(self.query_pool.clone(), 1, PipelineStage::ComputeShader)
                .unwrap()
        };

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }

    fn execution_time(&self) -> f32 {
        let mut query_results = [0u64; 2];
        self.query_pool
            .get_results(0..2, &mut query_results, QueryResultFlags::WAIT)
            .unwrap();

        (query_results[1] - query_results[0]) as f32
            * self
                .queue
                .device()
                .physical_device()
                .properties()
                .timestamp_period
            / 1000000.0
    }

    fn layout(&self) -> vulkano::memory::allocator::DeviceLayout {
        cs::DistanceField::LAYOUT
            .layout_for_len(glam::UVec3::splat(self.chunk_size).element_product() as u64)
            .unwrap()
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/distance_field/brute_force.comp"
    }
}
