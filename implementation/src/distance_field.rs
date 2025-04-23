use std::sync::Arc;

use glam::{UVec3, Vec3};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
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

pub struct DistanceFieldPipeline {
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    distance_field_allocator: SubbufferAllocator,
    query_pool: Arc<QueryPool>,

    pipeline: Arc<ComputePipeline>,
    pp_distance_field: Subbuffer<[u8]>,
    configuration_buffer: Subbuffer<cs::Configuration>,
    convergence_buffer: Subbuffer<cs::Convergence>,

    chunk_size: u32,
    focus_area_min: Vec3,
    focus_area_max: Vec3,
}

impl DistanceFieldPipeline {
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        chunk_size: u32,
        focal_point: UVec3,
        focus_size: u32,
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

        let distance_field_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        );

        let pp_distance_field = distance_field_allocator
            .allocate(
                cs::DistanceFieldA::LAYOUT
                    .layout_for_len(glam::UVec3::splat(chunk_size).element_product() as u64)
                    .unwrap(),
            )
            .unwrap();

        let configuration_buffer = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let convergence_buffer = Buffer::new_sized(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        Self {
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            distance_field_allocator,
            query_pool,

            pipeline,
            pp_distance_field,
            configuration_buffer,
            convergence_buffer,

            chunk_size,
            focus_area_min: focal_point
                .saturating_sub(glam::UVec3::splat(focus_size))
                .as_vec3(),
            focus_area_max: (focal_point + focus_size).as_vec3(),
        }
    }

    /// Computes the distance field. Selectively decides what combination of algorithms to use (JFA, and FIM).
    ///
    /// Returns the execution time of the compute shader in milliseconds.
    pub fn compute(&self, distance_field: Subbuffer<[u8]>, chunk: &Chunk, chunk_pos: Vec3) -> f32 {
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
        let descriptor_set_0 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, chunk.voxels.clone()),
                WriteDescriptorSet::buffer(1, distance_field),
                WriteDescriptorSet::buffer(2, self.pp_distance_field.clone()),
            ],
            [],
        )
        .unwrap();

        {
            let mut buffer = self.configuration_buffer.write().unwrap();

            let num_passes = (chunk.size.max_element() as f32).log2().ceil() as i32;
            buffer.jfa_initial_step_size = 2_i32.pow(num_passes as u32 - 1).into();
            buffer.jfa_num_passes = num_passes.into();

            buffer.focus_area_min = self.focus_area_min.to_array().into();
            buffer.focus_area_max = self.focus_area_max.to_array();
        }

        let layout = &self.pipeline.layout().set_layouts()[1];
        let descriptor_set_1 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.configuration_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.convergence_buffer.clone()),
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
                (descriptor_set_0, descriptor_set_1),
            )
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                cs::PushConstants {
                    chunk_size: chunk.size.into(),
                    max_dimension: chunk.size.max_element(),
                    chunk_pos: chunk_pos.into(),
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
        finished.then_signal_fence_and_flush().unwrap().boxed();

        self.execution_time()
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

    /// Allocates a device-only subbuffer to store a distance field of size `chunk_size^3`.
    pub fn allocate_distance_field(&self) -> Subbuffer<[u8]> {
        let layout = cs::DistanceFieldA::LAYOUT
            .layout_for_len(glam::UVec3::splat(self.chunk_size).element_product() as u64)
            .unwrap();
        self.distance_field_allocator.allocate(layout).unwrap()
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/distance_field/combined.comp"
    }
}
