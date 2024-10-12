use app::VoxelApp;
use vulkano::{image::ImageUsage, sync::GpuFuture};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

mod app;
mod ray_marcher_compute;
mod screen_quad_pipeline;
mod utils;

fn main() {
    let event_loop = EventLoop::new();
    let context = VulkanoContext::new(VulkanoConfig::default());
    let mut windows = VulkanoWindows::default();
    let _id = windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            title: "Voxels".to_string(),
            present_mode: vulkano::swapchain::PresentMode::Fifo,
            ..Default::default()
        },
        |_| {},
    );

    let render_target_id = 0;
    let primary_window_renderer = windows.get_primary_renderer_mut().unwrap();

    primary_window_renderer.add_additional_image_view(
        render_target_id,
        DEFAULT_IMAGE_FORMAT,
        ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
    );

    let graphics_queue = context.graphics_queue();

    let mut app = VoxelApp::new(
        graphics_queue.clone(),
        primary_window_renderer.swapchain_format(),
    );

    event_loop.run(move |event, _, control_flow| {
        let renderer = windows.get_primary_renderer_mut().unwrap();
        if process_event(renderer, render_target_id, &event, &mut app) {
            *control_flow = ControlFlow::Exit;
            return;
        }
    });
}

fn process_event(
    renderer: &mut VulkanoWindowRenderer,
    render_target_id: usize,
    event: &Event<()>,
    app: &mut VoxelApp,
) -> bool {
    match &event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => return true,
        Event::WindowEvent {
            event: WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. },
            ..
        } => renderer.resize(),
        Event::MainEventsCleared => {
            renderer.window().request_redraw();
        }
        Event::RedrawRequested(_) => 'redraw: {
            match renderer.window_size() {
                [w, h] => {
                    if w == 0.0 || h == 0.0 {
                        break 'redraw;
                    }
                }
            }

            let before_pipeline_future = match renderer.acquire() {
                Err(e) => {
                    eprintln!("Failed to acquire swapchain image: {:?}", e);
                    break 'redraw;
                }
                Ok(future) => future,
            };

            let image = renderer.get_additional_image_view(render_target_id);

            let after_compute = app
                .ray_marcher_pipeline
                .compute(image.clone())
                .join(before_pipeline_future);

            let after_render_pass_future = app.screen_quad_render_pass.render(
                after_compute,
                image,
                renderer.swapchain_image_view(),
            );

            renderer.present(after_render_pass_future, true);
        }
        _ => (),
    }

    false
}
