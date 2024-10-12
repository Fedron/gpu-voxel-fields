use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    format::Format,
    memory::allocator::StandardMemoryAllocator,
};

use crate::{
    ray_marcher_compute::RayMarcherComputePipeline, screen_quad_pipeline::ScreenQuadRenderPass,
};

pub struct VoxelApp {
    pub screen_quad_render_pass: ScreenQuadRenderPass,
    pub ray_marcher_pipeline: RayMarcherComputePipeline,
}

impl VoxelApp {
    pub fn new(graphics_queue: Arc<Queue>, swapchain_format: Format) -> Self {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            graphics_queue.device().clone(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            graphics_queue.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            graphics_queue.device().clone(),
            Default::default(),
        ));

        Self {
            screen_quad_render_pass: ScreenQuadRenderPass::new(
                graphics_queue.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                swapchain_format,
            ),
            ray_marcher_pipeline: RayMarcherComputePipeline::new(
                graphics_queue,
                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
            ),
        }
    }
}
