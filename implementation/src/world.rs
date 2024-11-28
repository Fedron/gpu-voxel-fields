use glium::{
    glutin::surface::WindowSurface,
    texture::{buffer_texture::BufferTexture, UnsignedTexture3d},
    uniforms::{ImageUnit, ImageUnitError},
    Display,
};

use crate::ray::Ray;

#[derive(Clone, Copy)]
pub enum Voxel {
    Air = 0,
    Stone = 1,
}

pub struct World {
    size: glam::UVec3,
    voxels_texture: BufferTexture<u8>,
    distance_field_texture: UnsignedTexture3d,
    pub is_dirty: bool,
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
            is_dirty: false,
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
        if !self.is_in_bounds(position) {
            return;
        }

        let index = self.position_to_index(position);
        let mut voxels = self.voxels_texture.map_write();
        voxels.set(index, voxel as u8);

        self.is_dirty = true;
    }

    pub fn is_voxel_hit(&self, mut ray: Ray) -> Hit {
        let hit = ray.intersect_aabb(glam::Vec3::ZERO, self.size.as_vec3());
        if !hit.does_intersect {
            return Hit {
                does_intersect: false,
                voxel_position: None,
                face_normal: None,
            };
        }

        ray.advance(hit.near);

        let voxels = self
            .voxels_texture
            .read()
            .expect("to read voxels buffer texture");

        let mut voxel_position = ray
            .position
            .clamp(glam::Vec3::ZERO, (self.size - 1).as_vec3());
        let mut face_normal = glam::Vec3::ZERO;

        let delta_dist = ray.inverse_direction.abs();
        let ray_step = ray.direction.signum();
        let mut side_dist = (ray.direction.signum() * (voxel_position - ray.position)
            + (ray.direction.signum() * 0.5)
            + 0.5)
            * ray.inverse_direction.abs();
        while self.is_in_bounds(voxel_position.as_uvec3()) {
            match voxels.get(self.position_to_index(voxel_position.as_uvec3())) {
                Some(&voxel) => {
                    if voxel != Voxel::Air as u8 {
                        return Hit {
                            does_intersect: true,
                            voxel_position: Some(voxel_position.as_uvec3()),
                            face_normal: Some(face_normal.as_ivec3()),
                        };
                    }
                }
                None => {
                    return Hit {
                        does_intersect: false,
                        voxel_position: None,
                        face_normal: None,
                    }
                }
            }

            if side_dist.x < side_dist.y {
                if side_dist.x < side_dist.z {
                    side_dist.x += delta_dist.x;
                    voxel_position.x += ray_step.x;
                    face_normal = glam::vec3(-ray_step.x, 0.0, 0.0);
                } else {
                    side_dist.z += delta_dist.z;
                    voxel_position.z += ray_step.z;
                    face_normal = glam::vec3(0.0, 0.0, -ray_step.z);
                }
            } else {
                if side_dist.y < side_dist.z {
                    side_dist.y += delta_dist.y;
                    voxel_position.y += ray_step.y;
                    face_normal = glam::vec3(0.0, -ray_step.y, 0.0);
                } else {
                    side_dist.z += delta_dist.z;
                    voxel_position.z += ray_step.z;
                    face_normal = glam::vec3(0.0, 0.0, -ray_step.z);
                }
            }
        }

        Hit {
            does_intersect: false,
            voxel_position: None,
            face_normal: None,
        }
    }

    fn position_to_index(&self, position: glam::UVec3) -> usize {
        (position.x + position.y * self.size.x + position.z * self.size.x * self.size.y) as usize
    }

    fn is_in_bounds(&self, position: glam::UVec3) -> bool {
        position.x < self.size.x && position.y < self.size.y && position.z < self.size.z
    }
}

pub struct Hit {
    pub does_intersect: bool,
    pub voxel_position: Option<glam::UVec3>,
    pub face_normal: Option<glam::IVec3>,
}
