use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, RenderPassBeginInfo,
        SecondaryAutoCommandBuffer, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

use crate::utils::create_entry_point;

#[repr(C)]
#[derive(BufferContents, Vertex)]
pub struct ScreenQuadVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub tex_coords: [f32; 2],
}

/// Returns the vertices and indices for a screen quad.
pub fn screen_quad_vertices() -> ([ScreenQuadVertex; 4], [u32; 6]) {
    (
        [
            ScreenQuadVertex {
                position: [-1.0, -1.0],
                tex_coords: [0.0, 1.0],
            },
            ScreenQuadVertex {
                position: [-1.0, 1.0],
                tex_coords: [0.0, 0.0],
            },
            ScreenQuadVertex {
                position: [1.0, 1.0],
                tex_coords: [1.0, 0.0],
            },
            ScreenQuadVertex {
                position: [1.0, -1.0],
                tex_coords: [1.0, 1.0],
            },
        ],
        [0, 2, 1, 0, 3, 2],
    )
}

/// A subpass graphics pipeline for rendering a screen quad.
pub struct ScreenQuadPipeline {
    graphics_queue: Arc<Queue>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,

    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    vertex_buffer: Subbuffer<[ScreenQuadVertex]>,
    index_buffer: Subbuffer<[u32]>,
}

impl ScreenQuadPipeline {
    pub fn new(
        graphics_queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let (vertices, indices) = screen_quad_vertices();
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();
        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        let pipeline = {
            let device = graphics_queue.device();
            let vs = create_entry_point(
                device.clone(),
                include_bytes!("shaders/screen_quad.vert.spv"),
            );
            let fs = create_entry_point(
                device.clone(),
                include_bytes!("shaders/screen_quad.frag.spv"),
            );

            let vertex_input_state = ScreenQuadVertex::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        Self {
            graphics_queue,
            subpass,
            pipeline,

            command_buffer_allocator,
            descriptor_set_allocator,

            vertex_buffer,
            index_buffer,
        }
    }

    /// Creates a descriptor set for the specified image, that can be used by this pipeline.
    pub fn create_descriptor_set(&self, image: Arc<ImageView>) -> Arc<PersistentDescriptorSet> {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let sampler = Sampler::new(
            self.graphics_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::sampler(0, sampler),
                WriteDescriptorSet::image_view(1, image),
            ],
            [],
        )
        .unwrap()
    }

    /// Draws the `image` over a screen quad.
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        image: Arc<ImageView>,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.as_ref(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();

        let desc_set = self.create_descriptor_set(image);
        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                desc_set,
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(self.index_buffer.clone())
            .unwrap()
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();

        builder.build().unwrap()
    }
}

/// A render pass for rendering a texture to a screen quad.
pub struct ScreenQuadRenderPass {
    graphics_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: ScreenQuadPipeline,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl ScreenQuadRenderPass {
    pub fn new(
        graphics_queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        output_format: Format,
    ) -> Self {
        let render_pass = vulkano::single_pass_renderpass!(
            graphics_queue.device().clone(),
            attachments: {
                color: {
                    format: output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            }
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pipeline = ScreenQuadPipeline::new(
            graphics_queue.clone(),
            subpass,
            memory_allocator,
            command_buffer_allocator.clone(),
            descriptor_set_allocator,
        );

        Self {
            graphics_queue,
            render_pass,
            pipeline,
            command_buffer_allocator,
        }
    }

    /// Renders the `view` to the `target` image.
    pub fn render<F>(
        &self,
        before_future: F,
        view: Arc<ImageView>,
        target: Arc<ImageView>,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        let image_dimensions: [u32; 2] = target.image().extent()[0..2].try_into().unwrap();

        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![target],
                ..Default::default()
            },
        )
        .unwrap();

        let mut command_buffer = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0; 4].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        let secondary_command_buffer = self.pipeline.draw(image_dimensions, view);

        command_buffer
            .execute_commands(secondary_command_buffer)
            .unwrap();
        command_buffer.end_render_pass(Default::default()).unwrap();

        let command_buffer = command_buffer.build().unwrap();

        let after_future = before_future
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }
}
