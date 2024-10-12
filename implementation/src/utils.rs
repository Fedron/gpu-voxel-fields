use std::sync::Arc;

use vulkano::{
    device::Device,
    shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo},
};

#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub min: glam::UVec3,
    pub max: glam::UVec3,
}

impl Bounds {
    pub fn contains(&self, position: glam::UVec3) -> bool {
        position.x >= self.min.x
            && position.y >= self.min.y
            && position.z >= self.min.z
            && position.x <= self.max.x
            && position.y <= self.max.y
            && position.z <= self.max.z
    }
}

pub fn create_entry_point(device: Arc<Device>, code: &[u8]) -> EntryPoint {
    let words = vulkano::shader::spirv::bytes_to_words(&code)
        .unwrap_or_else(|err| panic!("Failed to create shader {:?}", err))
        .into_owned();

    let module =
        unsafe { ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&words)).unwrap() };
    module.entry_point("main").unwrap()
}
