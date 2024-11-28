use glium::{
    glutin::surface::WindowSurface,
    texture::{buffer_texture::BufferTexture, UnsignedTexture3d},
    uniforms::{ImageUnit, ImageUnitError},
    Display,
};

#[derive(Clone, Copy)]
pub enum Voxel {
    Air = 0,
    Stone = 1,
}

pub struct World {
    size: glam::UVec3,
    voxels_texture: BufferTexture<u8>,
    distance_field_texture: UnsignedTexture3d,
}

impl World {
    pub fn new(display: &Display<WindowSurface>, size: glam::UVec3) -> Self {
        let voxels_texture: BufferTexture<u8> = BufferTexture::empty_dynamic(
            display,
            size.x as usize * size.y as usize * size.z as usize,
            glium::texture::buffer_texture::BufferTextureType::Unsigned,
        )
        .expect("to create buffer texture for voxel data");

        let distance_field_texture = UnsignedTexture3d::empty_with_format(
            display,
            glium::texture::UncompressedUintFormat::U8,
            glium::texture::MipmapsOption::NoMipmap,
            size.x,
            size.y,
            size.z,
        )
        .expect("to create texture for distance field");

        Self {
            size,
            voxels_texture,
            distance_field_texture,
        }
    }

    pub fn size(&self) -> glam::UVec3 {
        self.size
    }

    pub fn voxels_texture(&self) -> &BufferTexture<u8> {
        &self.voxels_texture
    }

    pub fn distance_field_image(&self) -> Result<ImageUnit<'_, UnsignedTexture3d>, ImageUnitError> {
        self.distance_field_texture
            .image_unit(glium::uniforms::ImageUnitFormat::R8UI)
    }

    pub fn set(&mut self, position: glam::UVec3, voxel: Voxel) {
        if position.x >= self.size.x || position.y >= self.size.y || position.z >= self.size.z {
            return;
        }

        let mut voxels = self.voxels_texture.map_write();
        voxels.set(
            (position.x + position.y * self.size.x + position.z * self.size.x * self.size.y)
                as usize,
            voxel as u8,
        );
    }
}
