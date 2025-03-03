use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
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
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    query_pool: Arc<QueryPool>,

    init_pipeline: Arc<ComputePipeline>,
    compute_pipeline: Arc<ComputePipeline>,
    pp_distance_field: Subbuffer<[u8]>,
}

impl DistanceFieldPipeline {
    /// Creates the distance field generation compute pipeline.
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        chunk_size: u32,
    ) -> Self {
        let init_pipeline = {
            let device = queue.device();
            let cs = init_cs::load(device.clone())
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
                ComputePipelineCreateInfo::new(stage, layout),
            )
            .unwrap()
        };

        let compute_pipeline = {
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
                ComputePipelineCreateInfo::new(stage, layout),
            )
            .unwrap()
        };

        let query_pool = QueryPool::new(
            queue.device().clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::new(QueryType::Timestamp)
            },
        )
        .unwrap();

        let pp_distance_field = Buffer::new_slice::<u8>(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            glam::UVec3::splat(chunk_size).element_product() as u64 * 4,
        )
        .unwrap();

        Self {
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            query_pool,

            init_pipeline,
            compute_pipeline,
            pp_distance_field,
        }
    }

    /// Calculates the distance field of a [`Chunk`] and writes distance information to a buffer.
    ///
    /// # Safety
    /// It is assumed the distance field buffer is of sufficient size to store the chunk.
    pub fn compute(&self, distance_field: Subbuffer<[u8]>, chunk: &Chunk) -> Box<dyn GpuFuture> {
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

        self.dispatch_init(&mut builder, distance_field.clone(), chunk);
        self.dispatch_compute(&mut builder, distance_field, chunk);

        unsafe {
            builder
                .write_timestamp(self.query_pool.clone(), 1, PipelineStage::ComputeShader)
                .unwrap()
        };

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }

    fn dispatch_init<'a, 'b>(
        &'a self,
        builder: &'b mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        distance_field: Subbuffer<[u8]>,
        chunk: &Chunk,
    ) -> &'b mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    where
        'a: 'b,
    {
        let layout = &self.init_pipeline.layout().set_layouts()[0];
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
            .bind_pipeline_compute(self.init_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.init_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.init_pipeline.layout().clone(),
                0,
                init_cs::PushConstants {
                    chunk_size: chunk.size.into(),
                },
            )
            .unwrap();

        unsafe {
            builder
                .dispatch(chunk.size.saturating_div(glam::UVec3::splat(8)).into())
                .unwrap()
        }
    }

    fn dispatch_compute<'a, 'b>(
        &'a self,
        builder: &'b mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        distance_field: Subbuffer<[u8]>,
        chunk: &Chunk,
    ) -> &'b mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>
    where
        'a: 'b,
    {
        let layout = &self.compute_pipeline.layout().set_layouts()[0];
        let ping_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, distance_field.clone()),
                WriteDescriptorSet::buffer(1, self.pp_distance_field.clone()),
            ],
            [],
        )
        .unwrap();
        let pong_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.pp_distance_field.clone()),
                WriteDescriptorSet::buffer(1, distance_field.clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .unwrap();

        let num_passes = (chunk.size.max_element() as f32).log2().ceil() as u32;
        for i in 0..num_passes {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.compute_pipeline.layout().clone(),
                    0,
                    if i % 2 == 0 {
                        ping_descriptor_set.clone()
                    } else {
                        pong_descriptor_set.clone()
                    },
                )
                .unwrap();

            let step_size = 1 << (num_passes - i - 1);
            builder
                .push_constants(
                    self.compute_pipeline.layout().clone(),
                    0,
                    cs::PushConstants {
                        chunk_size: chunk.size.into(),
                        step_size,
                    },
                )
                .unwrap();

            unsafe {
                builder
                    .dispatch(chunk.size.saturating_div(glam::UVec3::splat(8)).into())
                    .unwrap()
            };
        }

        if num_passes % 2 != 0 {
            builder
                .copy_buffer(CopyBufferInfo::new(
                    self.pp_distance_field.clone(),
                    distance_field.clone(),
                ))
                .unwrap();
        }

        builder
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

        layout (set = 0, binding = 0) buffer CurrentDistanceField {
            uint current_distance_field[];
        };
        layout (set = 0, binding = 1) buffer NextDistanceField {
            uint next_distance_field[];
        };

        layout (push_constant) uniform PushConstants {
            uvec3 chunk_size;
            int step_size;
        };

        uint get_index(uvec3 position) {
            return position.x + position.y * chunk_size.x + position.z * chunk_size.x * chunk_size.y;
        }

        bool is_valid_position(ivec3 position) {
            return all(greaterThanEqual(position, ivec3(0))) && all(lessThan(position, chunk_size));
        }

        uint unpack_distance(uint value) {
            return (value >> 8) & 0xFFu;
        }

        uint pack_distance(uint value, uint r, uint g, uint b) {
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
            uint min_distance = unpack_distance(current_distance_field[voxel_index]);
            if (min_distance == 0) {
                next_distance_field[voxel_index] = current_distance_field[voxel_index];
                return;
            }

            const ivec3 neighbours[6] = {
                ivec3(-1, 0, 0), ivec3(1, 0, 0),
                ivec3(0, -1, 0), ivec3(0, 1, 0),
                ivec3(0, 0, -1), ivec3(0, 0, 1),
            };

            for (int i = 0; i < 6; i++) {
                ivec3 step = neighbours[i] * step_size;
                ivec3 neighbour_pos = voxel_pos + step;
                if (!is_valid_position(neighbour_pos)) continue;

                uint neighbour_index = get_index(neighbour_pos);
                uint neighbour_distance = unpack_distance(current_distance_field[neighbour_index]);
                min_distance = min(min_distance, neighbour_distance + step_size);
            }

            next_distance_field[voxel_index] = (next_distance_field[voxel_index] & 0x00FFu) | (min_distance << 8);
        }",
    }
}

pub mod init_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
        #version 460

        layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

        layout (set = 0, binding = 0) buffer World {
            uint voxels[];
        };
        layout (set = 0, binding = 1) buffer DistanceField {
            uint distance_field[];
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

        uint pack_distance(uint value, uint r, uint g, uint b) {
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
            if (voxels[voxel_index] == 0) {
                distance_field[voxel_index] = pack_distance(max(max(chunk_size.x, chunk_size.y), chunk_size.z), 0, 0, 0);
            } else if (voxels[voxel_index] == 1) {
                distance_field[voxel_index] = pack_distance(0, 5, 5, 2);
            } else if (voxels[voxel_index] == 2) {
                distance_field[voxel_index] = pack_distance(0, 6, 6, 2);
            } else if (voxels[voxel_index] == 3) {
                distance_field[voxel_index] = pack_distance(0, 2, 6, 3);
            }
        }"
    }
}
