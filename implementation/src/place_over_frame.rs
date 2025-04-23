//! Graphics pipeline for drawing a frame to the screen.
//!
//! Taken from the Vulkano "interactive-fractal" example. Adapted to render a crosshair on-top of the final image and
//! render an `egui` gui.
//!
//! https://github.com/vulkano-rs/vulkano/blob/23606f05825adf5212f104ead9e95f9d325db1aa/examples/interactive-fractal/place_over_frame.rs

use crate::{crosshair_pipeline::CrosshairPipeline, pixels_draw_pipeline::PixelsDrawPipeline};
use egui_winit_vulkano::Gui;
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    format::Format,
    image::{view::ImageView, SampleCount},
    memory::allocator::StandardMemoryAllocator,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

/// A render pass which places an incoming image over frame filling it.
pub struct RenderPassPlaceOverFrame {
    gfx_queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pixels_draw_pipeline: PixelsDrawPipeline,
    crosshair_pipeline: CrosshairPipeline,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

impl RenderPassPlaceOverFrame {
    pub fn new(
        gfx_queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        output_format: Format,
        swapchain_image_views: &[Arc<ImageView>],
    ) -> RenderPassPlaceOverFrame {
        let render_pass = vulkano::ordered_passes_renderpass!(
            gfx_queue.device().clone(),
            attachments: {
                color: {
                    format: output_format,
                    samples: SampleCount::Sample1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            passes: [
                { color: [color], depth_stencil: {}, input: [] },
                { color: [color], depth_stencil: {}, input: [] }
            ]
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let pixels_draw_pipeline = PixelsDrawPipeline::new(
            gfx_queue.clone(),
            subpass.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator,
        );

        let crosshair_pipeline = CrosshairPipeline::new(
            gfx_queue.clone(),
            subpass,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
        );

        RenderPassPlaceOverFrame {
            gfx_queue,
            render_pass: render_pass.clone(),
            pixels_draw_pipeline,
            crosshair_pipeline,
            command_buffer_allocator,
            framebuffers: create_framebuffers(swapchain_image_views, render_pass),
        }
    }

    pub fn gui_subpass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 1).unwrap()
    }

    /// Places the view exactly over the target swapchain image. The texture draw pipeline uses a
    /// quad onto which it places the view.
    pub fn render<F>(
        &self,
        before_future: F,
        view: Arc<ImageView>,
        target: Arc<ImageView>,
        image_index: u32,
        gui: &mut Gui,
    ) -> Box<dyn GpuFuture>
    where
        F: GpuFuture + 'static,
    {
        // Get dimensions.
        let img_dims: [u32; 2] = target.image().extent()[0..2].try_into().unwrap();

        // Create primary command buffer builder.
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Begin render pass.
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0; 4].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        // Create secondary command buffer from texture pipeline & send draw commands.
        let cb = self.pixels_draw_pipeline.draw(img_dims, view);

        // Execute above commands (subpass).
        command_buffer_builder.execute_commands(cb).unwrap();

        let cb = self.crosshair_pipeline.draw(img_dims);
        command_buffer_builder.execute_commands(cb).unwrap();

        command_buffer_builder
            .next_subpass(
                Default::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();
        let cb = gui.draw_on_subpass_image(img_dims);
        command_buffer_builder.execute_commands(cb).unwrap();

        // End render pass.
        command_buffer_builder
            .end_render_pass(Default::default())
            .unwrap();

        // Build command buffer.
        let command_buffer = command_buffer_builder.build().unwrap();

        // Execute primary command buffer.
        let after_future = before_future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }

    pub fn recreate_framebuffers(&mut self, swapchain_image_views: &[Arc<ImageView>]) {
        self.framebuffers = create_framebuffers(swapchain_image_views, self.render_pass.clone());
    }
}

fn create_framebuffers(
    swapchain_image_views: &[Arc<ImageView>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    swapchain_image_views
        .iter()
        .map(|swapchain_image_view| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![swapchain_image_view.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
