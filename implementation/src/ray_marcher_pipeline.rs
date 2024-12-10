use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

use crate::world::World;

/// Compute pipeline to ray march a discrete distance field.
pub struct RayMarcherPipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    local_size: (u32, u32),

    camera_buffer: Subbuffer<cs::Camera>,
    world_buffer: Subbuffer<cs::World>,
}

impl RayMarcherPipeline {
    /// Creates a new ray marching compute pipeline.
    ///
    /// The pipeline will be bound to a specific [`World`] size.
    pub fn new<const X: usize, const Y: usize, const Z: usize>(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        world: &World<X, Y, Z>,
    ) -> Self
    where
        [(); X * Y * Z]:,
    {
        let local_size = match queue.device().physical_device().properties().subgroup_size {
            Some(subgroup_size) => (subgroup_size, subgroup_size),
            None => (8, 8),
        };

        let pipeline = {
            let device = queue.device();
            let cs = cs::load(device.clone())
                .unwrap()
                .specialize(
                    [(1, local_size.0.into()), (2, local_size.1.into())]
                        .into_iter()
                        .collect(),
                )
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

        let camera_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            cs::Camera {
                position: [0.0, 0.0, 0.0].into(),
                inverse_view: [[0.0; 4]; 4],
                inverse_projection: [[0.0; 4]; 4],
            },
        )
        .unwrap();

        let world_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            world.into(),
        )
        .unwrap();

        Self {
            queue,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
            local_size,

            camera_buffer,
            world_buffer,
        }
    }

    /// Ray marches through a discrete distance field, storing the output in `image_view`.
    pub fn compute(
        &self,
        image_view: Arc<ImageView>,
        distance_field: Arc<ImageView>,
        camera: cs::Camera,
    ) -> Box<dyn GpuFuture> {
        let image_extent = image_view.image().extent();
        let set_layouts = &self.pipeline.layout().set_layouts();

        let descriptor_set_0 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layouts[0].clone(),
            [
                WriteDescriptorSet::image_view(0, distance_field),
                WriteDescriptorSet::image_view(1, image_view),
            ],
            [],
        )
        .unwrap();

        {
            let mut camera_buffer = self.camera_buffer.write().unwrap();
            camera_buffer.inverse_projection = camera.inverse_projection;
            camera_buffer.inverse_view = camera.inverse_view;
            camera_buffer.position = camera.position;
        }

        let descriptor_set_1 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layouts[1].clone(),
            [
                WriteDescriptorSet::buffer(0, self.camera_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.world_buffer.clone()),
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

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                (descriptor_set_0, descriptor_set_1),
            )
            .unwrap();

        unsafe {
            builder.dispatch([
                (image_extent[0] + self.local_size.0 - 1) / self.local_size.0,
                (image_extent[1] + self.local_size.1 - 1) / self.local_size.1,
                1,
            ])
        }
        .unwrap();

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

        layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z = 1) in;

        layout (set = 0, binding = 0, r16ui) readonly uniform uimage3D distance_field;
        layout (set = 0, binding = 1, rgba8) uniform image2D output_image;

        layout (set = 1, binding = 0) buffer Camera {
            vec3 position;
            mat4 inverse_view;
            mat4 inverse_projection;
        } camera;

        layout (set = 1, binding = 1) buffer World {
            vec3 min;
            vec3 max;
            ivec3 size;
        } world;

        float compute_ao(ivec3 voxel_pos) {
            int neighbor_count = 0;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        ivec3 neighbour_pos = voxel_pos + ivec3(dx, dy, dz);
                        if (all(greaterThanEqual(neighbour_pos, ivec3(0))) && 
                            all(lessThan(neighbour_pos, world.size))) {
                            if (((imageLoad(distance_field, neighbour_pos).r >> 8) & 0xFFu) == 0) {
                                neighbor_count++;
                            }
                        }
                    }
                }
            }

            return 1.0 - float(neighbor_count) / 27.0;
        }

        vec4 sky_color(int image_height) {
            return vec4(mix(
                vec3(0.71, 0.85, 0.90),
                vec3(0.0, 0.45, 0.74),
                gl_GlobalInvocationID.y / image_height
            ), 1.0);
        }

        void unpack_r16_uint(uint packed, out uint value, out vec3 rgb332) {
            value = (packed >> 8) & 0xFFu;

            uint rgb = packed & 0xFFu;
            uint r = (rgb >> 5) & 0x7u;
            uint g = (rgb >> 2) & 0x7u;
            uint b = rgb & 0x3u;

            rgb332 = vec3(float(r) / 7.0, float(g) / 7.0, float(b) / 3.0);
        }

        void main() {
            ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
            ivec2 image_size = imageSize(output_image);

            if (any(greaterThanEqual(pixel_coord, image_size))) {
                return;
            }

            vec2 ndc = (vec2(pixel_coord) / vec2(image_size)) * 2.0 - 1.0;

            vec4 clip = vec4(ndc, 0.0, 1.0);
            vec4 eye = camera.inverse_projection * clip;
            eye = vec4(eye.xy, -1.0, 0.0);

            vec3 ray_dir = normalize((camera.inverse_view * eye).xyz);
            vec3 ray_origin = camera.position;

            vec3 voxel_size = (world.max - world.min) / vec3(world.size);

            vec3 inv_dir = 1.0 / ray_dir;
            vec3 t_min = (world.min - ray_origin) * inv_dir;
            vec3 t_max = (world.max - ray_origin) * inv_dir;

            vec3 t_enter = min(t_min, t_max);
            vec3 t_exit = max(t_min, t_max);

            float t_grid_enter = max(max(t_enter.x, t_enter.y), t_enter.z);
            float t_grid_exit = min(min(t_exit.x, t_exit.y), t_exit.z);

            if (t_grid_enter > t_grid_exit || t_grid_exit < 0.0) {
                imageStore(output_image, pixel_coord, sky_color(image_size.y));
                return;
            }

            ray_origin += max(t_grid_enter, 0.0) * ray_dir;

            ivec3 voxel_pos = ivec3(floor((ray_origin - world.min) / voxel_size));
            voxel_pos = clamp(voxel_pos, ivec3(0), world.size - 1);

            vec3 t_delta;
            for (int i = 0; i < 3; ++i) {
                if (ray_dir[i] > 0) {
                    t_max[i] = ((voxel_pos[i] + 1) * voxel_size[i] + world.min[i] - ray_origin[i]) / ray_dir[i];
                    t_delta[i] = voxel_size[i] / ray_dir[i];
                } else if (ray_dir[i] < 0) {
                    t_max[i] = (voxel_pos[i] * voxel_size[i] + world.min[i] - ray_origin[i]) / ray_dir[i];
                    t_delta[i] = -voxel_size[i] / ray_dir[i];
                } else {
                    t_max[i] = 1e30;
                    t_delta[i] = 1e30;
                }
            }

            for (int step = 0; step < 256; ++step) {
                uint df = imageLoad(distance_field, voxel_pos).r;
                uint voxel;
                vec3 voxel_rgb;
                unpack_r16_uint(df, voxel, voxel_rgb);

                if (voxel == 0) {
                    float ao = compute_ao(voxel_pos);
                    vec4 base_color = vec4(voxel_rgb, 1.0);
                    imageStore(output_image, pixel_coord, vec4(base_color.rgb * mix(1.0, ao, 0.2), base_color.a));
                    return;
                }

                if (t_max.x < t_max.y) {
                    if (t_max.x < t_max.z) {
                        voxel_pos.x += int(sign(ray_dir.x));
                        t_max.x += t_delta.x;
                    } else {
                        voxel_pos.z += int(sign(ray_dir.z));
                        t_max.z += t_delta.z;
                    }
                } else {
                    if (t_max.y < t_max.z) {
                        voxel_pos.y += int(sign(ray_dir.y));
                        t_max.y += t_delta.y;
                    } else {
                        voxel_pos.z += int(sign(ray_dir.z));
                        t_max.z += t_delta.z;
                    }
                }

                if (any(lessThan(voxel_pos, ivec3(0))) || any(greaterThanEqual(voxel_pos, world.size))) {
                    break;
                }
            }

            imageStore(output_image, pixel_coord, sky_color(image_size.y));
        }",
    }
}
