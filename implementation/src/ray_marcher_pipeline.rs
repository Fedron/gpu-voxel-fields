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
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        world: &World,
    ) -> Self {
        let local_size = match queue.device().physical_device().properties().subgroup_size {
            Some(subgroup_size) => (subgroup_size, subgroup_size),
            None => (8, 8),
        };

        let pipeline = {
            let device = queue.device();
            let cs = cs::load(device.clone())
                .unwrap()
                .specialize(
                    [
                        (0, (world.chunks.len() as u32).into()),
                        (1, local_size.0.into()),
                        (2, local_size.1.into()),
                    ]
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
        distance_fields: Vec<Subbuffer<[u8]>>,
        camera: cs::Camera,
    ) -> Box<dyn GpuFuture> {
        let image_extent = image_view.image().extent();
        let set_layouts = &self.pipeline.layout().set_layouts();

        let descriptor_set_0 = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            set_layouts[0].clone(),
            [
                WriteDescriptorSet::buffer_array(0, 0, distance_fields),
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

        layout(constant_id = 0) const uint num_distance_fields = 4096;

        layout (set = 0, binding = 0) buffer DistanceField {
            uint values[];
        } distance_fields[num_distance_fields];
        layout (set = 0, binding = 1, rgba8) uniform image2D output_image;

        layout (set = 1, binding = 0) buffer Camera {
            vec3 position;
            mat4 inverse_view;
            mat4 inverse_projection;
        } camera;

        layout (set = 1, binding = 1) buffer World {
            uvec3 size;
            uvec3 chunk_size;
            uvec3 num_chunks;
        } world;

        uint get_voxel(uvec3 position) {
            if (any(greaterThanEqual(position, world.size)))
                return -1;

            uvec3 chunk = position / world.chunk_size;
            uint chunk_index = chunk.x + chunk.y * world.num_chunks.x + chunk.z * world.num_chunks.x * world.num_chunks.z;

            uvec3 local_pos = position % world.chunk_size;
            uint index = local_pos.x + local_pos.y * world.chunk_size.x + local_pos.z * world.chunk_size.x * world.chunk_size.y;
            return distance_fields[chunk_index].values[index];
        }

        float compute_ao(ivec3 voxel_pos) {
            int neighbor_count = 0;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        ivec3 neighbour_pos = voxel_pos + ivec3(dx, dy, dz);
                        if (all(greaterThanEqual(neighbour_pos, ivec3(0))) && 
                            all(lessThan(neighbour_pos, world.size))) {
                            if (((get_voxel(neighbour_pos) >> 8) & 0xFFu) == 0) {
                                neighbor_count++;
                            }
                        }
                    }
                }
            }

            return 1.0 - float(neighbor_count) / 27.0;
        }

        vec4 sky_color(vec3 world_dir, int image_height) {
            float t = smoothstep(-0.1, 0.0, world_dir.y);
            vec3 sky = mix(vec3(0.714, 1.0, 1.0), vec3(0.286, 0.714, 1.0), t);

            float sun_intensity = max(0.0, dot(world_dir, normalize(vec3(1.0, 1.0, 1.0))));
            sun_intensity = smoothstep(0.999, 1.0, sun_intensity);
            vec3 final_color = mix(sky, vec3(1.0, 0.855, 0.0), sun_intensity);

            return vec4(final_color, 1.0);
        }

        void unpack_r16_uint(uint packed, out uint value, out vec3 rgb332) {
            value = (packed >> 8) & 0xFFu;

            uint rgb = packed & 0xFFu;
            uint r = (rgb >> 5) & 0x7u;
            uint g = (rgb >> 2) & 0x7u;
            uint b = rgb & 0x3u;

            rgb332 = vec3(float(r) / 7.0, float(g) / 7.0, float(b) / 3.0);
        }

        void get_starting_ray(vec2 pixel_coord, vec2 image_size, out vec3 ray_pos, out vec3 ray_dir) {
            vec2 ndc = (pixel_coord / image_size) * 2.0 - 1.0;

            vec4 clip = vec4(ndc, 0.0, 1.0);
            vec4 eye = camera.inverse_projection * clip;
            eye = vec4(eye.xy, -1.0, 0.0);

            ray_dir = normalize((camera.inverse_view * eye).xyz);
            ray_pos = camera.position;
        }

        bool intersect_aabb(vec3 ray_pos, vec3 inv_ray_dir, vec3 box_min, vec3 box_max, out vec2 intersection) {
            vec3 t_min = (box_min - ray_pos) * inv_ray_dir;
            vec3 t_max = (box_max - ray_pos) * inv_ray_dir;

            vec3 t_enter = min(t_min, t_max);
            vec3 t_exit = max(t_min, t_max);

            float t_near = max(max(t_enter.x, t_enter.y), t_enter.z);
            float t_far = min(min(t_exit.x, t_exit.y), t_exit.z);
            intersection = vec2(t_near, t_far);

            return t_far >= 0.0 && t_near <= t_far;
        }

        void main() {
            ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
            ivec2 image_size = imageSize(output_image);

            if (any(greaterThanEqual(pixel_coord, image_size))) {
                return;
            }

            vec3 ray_pos;
            vec3 ray_dir;
            get_starting_ray(pixel_coord, image_size, ray_pos, ray_dir);

            vec2 world_intersection;
            if (!intersect_aabb(ray_pos, 1.0 / ray_dir, vec3(0.0), world.size, world_intersection)) {
                imageStore(output_image, pixel_coord, sky_color(ray_dir, image_size.y));
                return;
            }

            ray_pos += max(0.0, world_intersection.x) * ray_dir;
            ray_pos += 0.001 * ray_dir;
            ivec3 voxel_pos = ivec3(floor(ray_pos));

            ivec3 step = ivec3(sign(ray_dir));
            vec3 t_max = (vec3(voxel_pos) + 0.5 + 0.5 * vec3(step) - ray_pos) / ray_dir;
            vec3 t_delta = abs(1.0 / ray_dir);

            vec4 final_color = sky_color(ray_dir, image_size.y);
            for (int i = 0; i < 1024; i++) {
                if (any(lessThan(voxel_pos, ivec3(0))) || any(greaterThanEqual(voxel_pos, world.size))) {
                    break;
                }

                uint voxel = get_voxel(voxel_pos);
                uint distance;
                vec3 voxel_rgb;
                unpack_r16_uint(voxel, distance, voxel_rgb);

                if (distance == 0) {
                    float ao = compute_ao(voxel_pos);
                    final_color = vec4(voxel_rgb * mix(1.0, ao, 0.2), 1.0);
                    break;
                }

                for (int j = 0; j < distance; j++) {
                    if (t_max.x < t_max.y && t_max.x < t_max.z) {
                        voxel_pos.x += step.x;
                        t_max.x += t_delta.x;
                    } else if (t_max.y < t_max.z) {
                        voxel_pos.y += step.y;
                        t_max.y += t_delta.y;
                    } else {
                        voxel_pos.z += step.z;
                        t_max.z += t_delta.z;
                    }
                }
            }

            imageStore(output_image, pixel_coord, final_color);
        }",
    }
}
