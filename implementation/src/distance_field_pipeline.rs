use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::view::ImageView,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

use crate::world::World;

/// Compute pipeline to calculate the discrete distance field for a [`World`].
pub struct DistanceFieldPipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl DistanceFieldPipeline {
    /// Creates the distance field generation compute pipeline.
    pub fn new(
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let pipeline = {
            let device = queue.device();
            let cs = cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
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
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    /// Calculates the distance field of a [`World`] and writes distance information to the red channel of an image.
    ///
    /// # Safety
    /// The underlying compute shader uses an unsigned integer image in the format `R8_UINT`.
    ///
    /// It is assumed the image is at least the size of the world.
    pub fn compute<const X: usize, const Y: usize, const Z: usize>(
        &self,
        image_view: Arc<ImageView>,
        world: &World<X, Y, Z>,
    ) -> Box<dyn GpuFuture>
    where
        [(); X * Y * Z]:,
    {
        let image_extent = image_view.image().extent();
        let layout = &self.pipeline.layout().set_layouts()[0];
        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, image_view),
                WriteDescriptorSet::buffer(1, world.voxels.clone()),
            ],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let push_constants = cs::PushConstants {
            world_size: [X as u32, Y as u32, Z as u32],
        };

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap();

        unsafe { builder.dispatch(image_extent) }.unwrap();

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();
        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
        #version 460

        layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

        layout (set = 0, binding = 0, r8ui) writeonly uniform uimage3D distance_field;
        layout (set = 0, binding = 1) buffer World {
            uint voxels[];
        } world;

        layout (push_constant) uniform PushConstants {
            uvec3 world_size;
        } push_constants;

        // Calculates the Chebyshev distance
        uint dist(uvec3 a, uvec3 b) {
            return uint(max(max(abs(b.x - a.x), abs(b.y - a.y)), abs(b.z - a.z)));
        }

        int get_voxel(uvec3 position) {
            if (any(greaterThanEqual(position, push_constants.world_size)))
                return -1;

            uint index = position.x + position.y * push_constants.world_size.x + position.z * push_constants.world_size.x * push_constants.world_size.y;
            return int(world.voxels[index]);
        }

        void main() {
            // Solid voxel, don't want to calculate distance
            int voxel = get_voxel(uvec3(gl_GlobalInvocationID.xyz));
            if (voxel > 0) {
                imageStore(distance_field, ivec3(gl_GlobalInvocationID.xyz), uvec4(0));
                return;
            }

            uint min_distance = push_constants.world_size.x * push_constants.world_size.y * push_constants.world_size.z;
            for (int x = 0; x < push_constants.world_size.x; x++) {
                for (int y = 0; y < push_constants.world_size.y; y++) {
                    for (int z = 0; z < push_constants.world_size.z; z++) {
                        if (get_voxel(uvec3(x, y, z)) > 0) {
                            uint neighbour_distance = dist(uvec3(x, y, z), uvec3(gl_GlobalInvocationID.xyz));
                            min_distance = min(min_distance, neighbour_distance);
                        }
                    }
                }
            }

            imageStore(distance_field, ivec3(gl_GlobalInvocationID.xyz), uvec4(min_distance));
        }",
    }
}
