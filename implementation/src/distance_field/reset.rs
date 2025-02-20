use std::sync::Arc;

use vulkano::{
    buffer::Subbuffer,
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

/// Compute pipeline to calculate the discrete distance field for a [`World`].
pub struct ResetDistanceFieldPipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    query_pool: Arc<QueryPool>,
}

impl ResetDistanceFieldPipeline {
    /// Creates the distance field generation compute pipeline.
    pub fn new(
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
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
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            query_pool,
        }
    }

    /// Calculates the distance field of a [`Gdutk`] and writes distance information to a buffer.
    ///
    /// # Safety
    /// It is assumed the distance field buffer is of sufficient size to store the chunk.
    pub fn compute(&self, distance_field: Subbuffer<[u8]>, chunk: &Chunk) -> Box<dyn GpuFuture> {
        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, distance_field),
                WriteDescriptorSet::buffer(1, chunk.voxels.clone()),
            ],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let push_constants = cs::PushConstants {
            chunk_size: chunk.size.into(),
        };

        unsafe {
            builder
                .reset_query_pool(self.query_pool.clone(), 0..2)
                .unwrap()
                .write_timestamp(self.query_pool.clone(), 0, PipelineStage::ComputeShader)
        }
        .unwrap()
        .bind_pipeline_compute(self.pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.pipeline.layout().clone(),
            0,
            descriptor_set,
        )
        .unwrap()
        .push_constants(self.pipeline.layout().clone(), 0, push_constants)
        .unwrap();

        unsafe {
            builder
                .dispatch(chunk.size.saturating_div(glam::UVec3::splat(8)).into())
                .unwrap()
                .write_timestamp(self.query_pool.clone(), 1, PipelineStage::ComputeShader)
        }
        .unwrap();

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }

    /// Gets the previous execution time of the compute shader.
    ///
    /// Will block until a result is available.
    pub fn execution_time(&self) -> f32 {
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
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
        #version 460

        layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

        layout (set = 0, binding = 0) buffer DistanceField {
            uint distance_field[];
        };
        layout (set = 0, binding = 1) buffer World {
            uint voxels[];
        };

        layout (push_constant) uniform PushConstants {
            uvec3 chunk_size;
        };

        uint get_index(uvec3 position) {
            return position.x + position.y * chunk_size.x + position.z * chunk_size.x * chunk_size.y;
        }

        bool is_valid_position(ivec3 position) {
            return all(greaterThanEqual(position, ivec3(0))) && all(lessThan(position, chunk_size));
        }

        uint pack_r16_uint(uint value, uint r, uint g, uint b) {
            value = value & 0xFFu;
            r = r & 0x7u;
            g = g & 0x7u;
            b = b & 0x3u;

            uint rgb = (r << 5) | (g << 2) | b;

            return (value << 8) | rgb;
        }

        void main() {
            ivec3 voxel_pos = ivec3(gl_GlobalInvocationID.xyz);
            if (!is_valid_position(voxel_pos)) {
                return;
            }

            uint voxel_index = get_index(voxel_pos);
            if (voxels[voxel_index] > 0) {
                uint value;
                if (voxels[voxel_index] == 1) { // Stone
                    distance_field[voxel_index] = pack_r16_uint(0, 5, 5, 2);
                } else if (voxels[voxel_index] == 2) { // Sand
                    distance_field[voxel_index] = pack_r16_uint(0, 6, 6, 2);
                } else if (voxels[voxel_index] == 3) { // Water
                    distance_field[voxel_index] = pack_r16_uint(0, 2, 6, 3);
                }
            } else {
                distance_field[voxel_index] = pack_r16_uint(max(max(chunk_size.x, chunk_size.y), chunk_size.z), 0, 0, 0);
            }
        }
        ",
    }
}
