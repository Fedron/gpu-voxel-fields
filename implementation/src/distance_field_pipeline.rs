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
pub struct DistanceFieldPipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    query_pool: Arc<QueryPool>,
}

impl DistanceFieldPipeline {
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
    pub fn compute(&self, distance_field: Subbuffer<[u32]>, chunk: &Chunk) -> Box<dyn GpuFuture> {
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
            uint values[];
        } distance_field;
        layout (set = 0, binding = 1) buffer World {
            uint voxels[];
        } chunk;

        layout (push_constant) uniform PushConstants {
            uvec3 chunk_size;
        } push_constants;

        uint dist(ivec3 a, ivec3 b) {
            return uint(abs(b.x - a.x) + abs(b.y - a.y) + abs(b.z - a.z));
        }

        uint get_index(uvec3 position) {
            return position.x + position.y * push_constants.chunk_size.x + position.z * push_constants.chunk_size.x * push_constants.chunk_size.y;
        }

        int get_voxel(ivec3 position) {
            if (any(lessThan(position, ivec3(0))) || any(greaterThanEqual(position, push_constants.chunk_size)))
                return 1;

            return int(chunk.voxels[get_index(uvec3(position))]);
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
            if (any(greaterThanEqual(uvec3(gl_GlobalInvocationID.xyz), push_constants.chunk_size))) {
                return;
            }

            int voxel = get_voxel(ivec3(gl_GlobalInvocationID.xyz));
            // Solid voxel, don't want to calculate distance
            if (voxel > 0) {
                uint value;
                if (voxel == 1) { // Stone
                    value = pack_r16_uint(0, 5, 5, 2);
                } else if (voxel == 2) { // Sand
                    value = pack_r16_uint(0, 6, 6, 2);
                } else if (voxel == 3) { // Water
                    value = pack_r16_uint(0, 2, 6, 3);
                } else if (voxel == 4) { // Sand Generator
                    value = pack_r16_uint(0, 5, 4, 1);
                } else if (voxel == 5) { // Water Generator
                    value = pack_r16_uint(0, 1, 2, 2);
                }

                distance_field.values[get_index(gl_GlobalInvocationID.xyz)] = value;
                return;
            }

            uint min_distance = max(max(push_constants.chunk_size.x, push_constants.chunk_size.y), push_constants.chunk_size.z);
            for (int x = -1; x <= int(push_constants.chunk_size.x); x++) {
                for (int y = -1; y <= int(push_constants.chunk_size.y); y++) {
                    for (int z = -1; z <= int(push_constants.chunk_size.z); z++) {
                        if (get_voxel(ivec3(x, y, z)) > 0) {
                            uint neighbour_distance = dist(ivec3(x, y, z), ivec3(gl_GlobalInvocationID.xyz));
                            min_distance = min(min_distance, neighbour_distance);
                        }
                    }
                }
            }

            distance_field.values[get_index(gl_GlobalInvocationID.xyz)] = pack_r16_uint(min_distance, 0, 0, 0);
        }",
    }
}
