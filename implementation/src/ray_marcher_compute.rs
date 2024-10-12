use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::view::ImageView,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

use crate::utils::create_entry_point;

/// A compute pipeline that performs ray marching.
pub struct RayMarcherComputePipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl RayMarcherComputePipeline {
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let pipeline = {
            let device = queue.device();
            let cs = create_entry_point(
                device.clone(),
                include_bytes!("shaders/ray_marcher.comp.spv"),
            );
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

        Self {
            queue,
            pipeline,

            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    /// Runs the ray marching compute pipeline, outputting to the given `image_view`.
    pub fn compute(&self, image_view: Arc<ImageView>) -> Box<dyn GpuFuture> {
        let image_extent = image_view.image().extent();
        let pipeline_layout = self.pipeline.layout();
        let descriptor_layout = pipeline_layout.set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            descriptor_layout.clone(),
            [WriteDescriptorSet::image_view(0, image_view)],
            [],
        )
        .unwrap();

        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .unwrap()
            .dispatch([image_extent[0] / 8, image_extent[1] / 8, 1])
            .unwrap();

        let command_buffer = command_buffer.build().unwrap();

        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}
