use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferInheritanceInfo, CommandBufferUsage, SecondaryAutoCommandBuffer,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex as VkVertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};

/// Draws a 2D crosshair in the center of the screen.
pub struct CrosshairPipeline {
    queue: Arc<Queue>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[Vertex]>,
}

impl CrosshairPipeline {
    /// Creates a new [`GraphicsPipeline`] to draw a static crosshair onto the `subpass`.
    pub fn new(
        queue: Arc<Queue>,
        subpass: Subpass,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> Self {
        let pipeline = {
            let device = queue.device();
            let vs = vs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
            let fs = fs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
            let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();
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
            [
                Vertex {
                    position: [-0.01, 0.0],
                },
                Vertex {
                    position: [0.01, 0.0],
                },
                Vertex {
                    position: [0.0, -0.01],
                },
                Vertex {
                    position: [0.0, 0.01],
                },
            ],
        )
        .unwrap();

        Self {
            queue,
            subpass,
            pipeline,
            command_buffer_allocator,
            vertex_buffer,
        }
    }

    /// Draws the crosshair.
    pub fn draw(&self, viewport_dimensions: [u32; 2]) -> Arc<SecondaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::secondary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();

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
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();
        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

        builder.build().unwrap()
    }
}

#[repr(C)]
#[derive(BufferContents, VkVertex)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(1.0, 1.0, 1.0, 1.0);
            }
        ",
    }
}
