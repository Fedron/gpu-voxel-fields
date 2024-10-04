use glium::{uniform, Surface};

fn main() {
    let event_loop = glium::winit::event_loop::EventLoop::builder()
        .build()
        .expect("event loop to be built");

    let (window, display) = glium::backend::glutin::SimpleWindowBuilder::new()
        .with_title("Dynamic Scene with an SVDAG")
        .with_inner_size(1280, 720)
        .build(&event_loop);

    let raymarched_texture = glium::Texture2d::empty_with_format(
        &display,
        glium::texture::UncompressedFloatFormat::U8U8U8U8,
        glium::texture::MipmapsOption::NoMipmap,
        1280,
        720,
    )
    .expect("to create texture");

    let raymarcher = glium::program::ComputeShader::from_source(
        &display,
        include_str!("shaders/raymarcher.comp"),
    )
    .expect("to create ray marcher compute shader");

    #[allow(deprecated)]
    event_loop
        .run(move |event, window_target| {
            match event {
                glium::winit::event::Event::WindowEvent { event, .. } => match event {
                    glium::winit::event::WindowEvent::CloseRequested => window_target.exit(),
                    glium::winit::event::WindowEvent::RedrawRequested => {
                        let mut frame = display.draw();
                        frame.clear_color(1.0, 0.0, 1.0, 1.0);

                        let out_image = raymarched_texture
                            .image_unit(glium::uniforms::ImageUnitFormat::RGBA8)
                            .expect("to create image unit")
                            .set_access(glium::uniforms::ImageUnitAccess::Write);
                        raymarcher.execute(
                            uniform! {
                                out_image: out_image,
                            },
                            1280,
                            720,
                            1,
                        );

                        raymarched_texture
                            .as_surface()
                            .fill(&frame, glium::uniforms::MagnifySamplerFilter::Nearest);

                        frame.finish().expect("to finish drawing");
                    }
                    glium::winit::event::WindowEvent::Resized(window_size) => {
                        display.resize(window_size.into());
                    }
                    _ => (),
                },
                glium::winit::event::Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => (),
            };
        })
        .unwrap();
}
