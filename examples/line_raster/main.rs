use anyhow::Result;
use ash::{prelude::VkResult, vk};
use bytemuck::{Pod, Zeroable};
use dolly::prelude::YawPitch;
use glam::{Mat4, Vec2, Vec3, Vec4, vec3};
use gpu_allocator::MemoryLocation;
use myndgera::{
    App, AppState, Camera, ComputeHandle, Device, FIXED_TIME_STEP, Framework, GpuBuffer,
    KeyboardMap, RenderContext, bytes_of, dispatch_optimal,
    math::{cos, erot, hash13, look_at, sin, smooth_floor},
    passes::{
        bloom::{Bloom, BloomParams},
        taa::{Taa, TaaParams},
    },
    vulkan::{
        Buffer, FragmentOutputDesc, FragmentShaderDesc, FrameGuard, ImageHandle, RenderHandle,
        ScreenRelation, VertexInputDesc, VertexShaderDesc, ViewTarget,
    },
};
use std::{error::Error, f32::consts::PI, sync::Arc};
use winit::event_loop::EventLoop;

const NUM_LIGHTS: usize = 4;
const NUM_RAYS: usize = 12500 * NUM_LIGHTS;
const NUM_BOUNCES: usize = 8;
const NUM_LINES: usize = NUM_RAYS * NUM_BOUNCES;

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable)]
struct Light {
    transform: Mat4,
    color: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Ray {
    color: Vec4,
    start: Vec3,
    end: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpawnPC {
    num_rays: u32,
    num_bounces: u32,
    time: f32,
    noise_offset: Vec2,
    line_buffer: u64,
    lights_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RasterPC {
    red_image: u32,
    green_image: u32,
    blue_image: u32,
    depth_image: u32,
    noise_offset: Vec2,
    camera_buffer: u64,
    line_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ResolvePC {
    target_image: u32,
    red_image: u32,
    green_image: u32,
    blue_image: u32,
    depth_image: u32,
    camera_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PostProcessPC {
    current_image: u32,
    hdr_sampled: u32,
    hdr_storage: u32,
}

struct Trig {
    lines_buffer: Buffer,
    lights_buffer: Buffer,
    spawn_pass: ComputeHandle,
    clear_pass: ComputeHandle,
    raster_pass: ComputeHandle,
    resolve_pass: ComputeHandle,
    postprocess_pass: RenderHandle,
    accumulate_images: Vec<ImageHandle>,
    view_target: ViewTarget,
    depth_image: ImageHandle,
    bloom: Bloom,
    taa: Taa,
    device: Arc<Device>,
}

impl Framework for Trig {
    fn name() -> &'static str {
        "Line Rasteriazation"
    }

    fn init(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        state.camera = Camera::new(vec3(0., 0., 10.), 0., 0.);

        let lines_buffer = ctx.device.create_buffer(
            GpuBuffer::<Ray, NUM_LINES>::SIZE as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        )?;
        let lights_buffer = ctx.device.create_buffer(
            GpuBuffer::<Light, NUM_LIGHTS>::SIZE as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<SpawnPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let spawn_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/spawn.comp.glsl",
            &[push_constant_range],
            &[state.texture_arena.sampled_set_layout],
        )?;

        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let clear_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/clear.comp.glsl",
            &[push_constant_range],
            &[state.texture_arena.storage_set_layout],
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let raster_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/raster.comp.glsl",
            &[push_constant_range],
            &[
                state.texture_arena.sampled_set_layout,
                state.texture_arena.storage_set_layout,
            ],
        )?;

        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<ResolvePC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let resolve_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/resolve.comp.glsl",
            &[push_constant_range],
            &[
                state.texture_arena.sampled_set_layout,
                state.texture_arena.storage_set_layout,
            ],
        )?;

        let vertex_shader_desc = VertexShaderDesc {
            shader_path: "shaders/screen_trig.vert".into(),
            ..Default::default()
        };
        let fragment_shader_desc = FragmentShaderDesc {
            shader_path: "examples/line_raster/postprocess.frag.glsl".into(),
            ..Default::default()
        };
        let fragment_output_desc = FragmentOutputDesc {
            surface_format: ctx.swapchain.format(),
            ..Default::default()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<PostProcessPC>() as _)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let postprocess_pass = state.pipeline_arena.create_render_pipeline(
            VertexInputDesc::default(),
            vertex_shader_desc,
            fragment_shader_desc,
            fragment_output_desc,
            &[push_constant_range],
            &[
                state.texture_arena.sampled_set_layout,
                state.texture_arena.storage_set_layout,
            ],
        )?;

        let mut accumulate_images = vec![];
        let image_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: ctx.swapchain.extent.width,
                height: ctx.swapchain.extent.height,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_UINT)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL);
        for i in 0..3 {
            let image = state.texture_arena.push_image(
                image_info,
                ScreenRelation::Identity,
                &[],
                Some(&format!("Ray Image {i}")),
            )?;
            accumulate_images.push(image);
        }
        let view_target = ViewTarget::new(ctx, state, vk::Format::B10G11R11_UFLOAT_PACK32)?;

        let depth_image = state.texture_arena.push_image(
            image_info.format(vk::Format::R16_SFLOAT),
            ScreenRelation::Identity,
            &[],
            Some("Depth Image"),
        )?;

        let bloom = Bloom::new(ctx, state)?;
        let taa = Taa::new(ctx, state)?;

        state.key_map = {
            use winit::keyboard::KeyCode::*;
            KeyboardMap::new()
                .bind(KeyW, ("move_fwd", 1.0))
                .bind(KeyS, ("move_fwd", -1.0))
                .bind(KeyD, ("move_right", 1.0))
                .bind(KeyA, ("move_right", -1.0))
                .bind(KeyQ, ("move_up", -1.0))
                .bind(KeyE, ("move_up", 1.0))
                .bind(ShiftLeft, ("boost", 1.0))
                .bind(ControlLeft, ("boost", -1.0))
        };

        Ok(Self {
            lights_buffer,
            lines_buffer,
            spawn_pass,
            clear_pass,
            raster_pass,
            resolve_pass,
            postprocess_pass,
            accumulate_images,
            view_target,
            depth_image,
            bloom,
            taa,
            device: ctx.device.clone(),
        })
    }

    fn draw(
        &mut self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &mut FrameGuard,
    ) -> VkResult<()> {
        let idx = frame.image_idx;
        let texture_arena = &mut state.texture_arena;

        let global_barrier = |src, dst| {
            let mem_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(src)
                .dst_stage_mask(dst);
            self.device.pipeline_barrier(
                frame.command_buffer(),
                &vk::DependencyInfo::default().memory_barriers(&[mem_barrier]),
            )
        };

        let raster_push_constant = RasterPC {
            red_image: texture_arena.get_storage_idx(self.accumulate_images[0], 0),
            green_image: texture_arena.get_storage_idx(self.accumulate_images[1], 0),
            blue_image: texture_arena.get_storage_idx(self.accumulate_images[2], 0),
            depth_image: texture_arena.get_storage_idx(self.depth_image, 0),
            noise_offset: rand::random::<Vec2>(),
            camera_buffer: state.camera_uniform_gpu.address,
            line_buffer: self.lines_buffer.address,
        };

        {
            let pipeline = state.pipeline_arena.get_pipeline(self.clear_pass);
            frame.bind_push_constants(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[raster_push_constant],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                &[texture_arena.storage_set],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
            const SUBGROUP_SIZE: u32 = 16;
            let extent = ctx.swapchain.extent();
            frame.dispatch(
                dispatch_optimal(extent.width, SUBGROUP_SIZE),
                dispatch_optimal(extent.height, SUBGROUP_SIZE),
                1,
            );
        }

        global_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        {
            let pipeline = state.pipeline_arena.get_pipeline(self.raster_pass);
            frame.bind_push_constants(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[raster_push_constant],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                &[texture_arena.sampled_set, texture_arena.storage_set],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
            frame.dispatch(dispatch_optimal(NUM_LINES as u32, 64), 1, 1);
        }

        global_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        {
            self.device.image_transition(
                frame.command_buffer(),
                &texture_arena
                    .get_image(*self.view_target.main_image())
                    .inner,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            let pipeline = state.pipeline_arena.get_pipeline(self.resolve_pass);
            frame.bind_push_constants(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[ResolvePC {
                    target_image: texture_arena.get_storage_idx(*self.view_target.main_image(), 0),
                    red_image: texture_arena.get_storage_idx(self.accumulate_images[0], 0),
                    green_image: texture_arena.get_storage_idx(self.accumulate_images[1], 0),
                    blue_image: texture_arena.get_storage_idx(self.accumulate_images[2], 0),
                    depth_image: texture_arena.get_storage_idx(self.depth_image, 0),
                    camera_buffer: state.camera_uniform_gpu.address,
                }],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                &[texture_arena.sampled_set, texture_arena.storage_set],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
            const SUBGROUP_SIZE: u32 = 16;
            let extent = ctx.swapchain.extent();
            frame.dispatch(
                dispatch_optimal(extent.width, SUBGROUP_SIZE),
                dispatch_optimal(extent.height, SUBGROUP_SIZE),
                1,
            );
        }

        global_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        self.taa.apply(
            ctx,
            state,
            frame,
            TaaParams {
                view_target: &self.view_target,
                depth_image: self.depth_image,
            },
        );

        global_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
        );

        self.bloom.apply(
            ctx,
            state,
            frame,
            BloomParams {
                target_image: *self.view_target.main_image(),
                target_current_layout: vk::ImageLayout::GENERAL,
                strength: 4. / 16.,
                width: 2.,
            },
        );

        global_barrier(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::PipelineStageFlags2::ALL_GRAPHICS,
        );

        {
            let texture_arena = &mut state.texture_arena;
            frame.begin_rendering(
                &ctx.swapchain.images[frame.image_idx],
                &ctx.swapchain.views[frame.image_idx],
                vk::AttachmentLoadOp::DONT_CARE,
                [0., 0.025, 0.025, 1.0],
            );
            let pipeline = state.pipeline_arena.get_pipeline(self.postprocess_pass);
            frame.bind_push_constants(
                pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &[PostProcessPC {
                    current_image: texture_arena.get_sampled_idx(state.swapchain_handles[idx], 0),
                    hdr_sampled: texture_arena.get_sampled_idx(*self.view_target.main_image(), 0),
                    hdr_storage: texture_arena.get_storage_idx(*self.view_target.main_image(), 0),
                }],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                &[
                    state.texture_arena.sampled_set,
                    state.texture_arena.storage_set,
                ],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pipeline.pipeline);
            frame.draw(3, 1, 0, 0);
            frame.end_rendering();
        }

        Ok(())
    }

    fn update(
        &mut self,
        ctx: &RenderContext,
        state: &mut AppState,
        cbuff: &vk::CommandBuffer,
    ) -> Result<()> {
        let time = state.time;
        let mut lights = [Light::default(); NUM_LIGHTS as usize];
        let make_light = |tr: Mat4, col: Vec3| Light {
            transform: tr,
            color: col.extend(0.),
        };
        let tr = |pos, i| {
            let i = i as f32;
            let coeffs = hash13(i + 33.42) * 2. - 1.;
            let t = smooth_floor(time * 0.15 + 20. + i * 2.5, 3.);
            let ax = coeffs * vec3(sin(-t), cos(t), sin(t + PI / 2.));
            let rot = Mat4::from_axis_angle(ax.normalize(), t);
            rot * Mat4::from_scale(Vec3::splat(3.))
                * look_at(pos, erot(-pos * 0.5, ax.normalize(), (t) + 0.5))
        };
        lights[0] = make_light(tr(vec3(1., 1.5, -1.5), 0), vec3(1., 1., 1.));
        lights[1] = make_light(tr(vec3(-1.5, 1., -1.), 1), vec3(1., 0., 0.));
        lights[2] = make_light(tr(vec3(1.5, -1., 1.), 2), vec3(1., 0., 0.));
        lights[3] = make_light(tr(vec3(1., -1., -1.5), 3), vec3(1., 1., 1.));

        let buffer_data = GpuBuffer {
            size: lights.len() as u32,
            data: lights,
        };
        state
            .staging_write
            .write_buffer(self.lights_buffer.buffer, bytes_of(&buffer_data));

        let extent = ctx.swapchain.extent();
        state.camera.jitter = self
            .taa
            .get_jitter(state.frame, extent.width, extent.height);

        let pipeline = state.pipeline_arena.get_pipeline(self.spawn_pass);
        let spawn_push_constant = SpawnPC {
            num_rays: NUM_RAYS as u32,
            num_bounces: NUM_BOUNCES as u32,
            time: state.time,
            noise_offset: rand::random::<Vec2>(),
            lights_buffer: self.lights_buffer.address,
            line_buffer: self.lines_buffer.address,
        };
        ctx.device.bind_descriptor_sets(
            &cbuff,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            &[state.texture_arena.sampled_set],
        );
        ctx.device.bind_push_constants(
            &cbuff,
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            bytes_of(&spawn_push_constant),
        );
        ctx.device
            .bind_pipeline(&cbuff, vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
        ctx.device
            .dispatch(&cbuff, dispatch_optimal(NUM_RAYS as u32, 256), 1, 1);

        if state.input.mouse_state.left_held() {
            let sensitivity = 0.5;
            state.camera.rig.driver_mut::<YawPitch>().rotate_yaw_pitch(
                -sensitivity * state.input.mouse_state.delta.x,
                -sensitivity * state.input.mouse_state.delta.y,
            );
        }

        let dt = FIXED_TIME_STEP as f32;
        let key_map = state.key_map.map(&state.input.keyboard_state);
        let translation = Vec3::new(
            key_map["move_right"],
            key_map["move_up"],
            -key_map["move_fwd"],
        );

        let rotation: glam::Quat = state.camera.rig.final_transform.rotation.into();
        let move_vec = rotation * translation.clamp_length_max(1.0) * 4.0f32.powf(key_map["boost"]);

        state
            .camera
            .rig
            .driver_mut::<dolly::drivers::Position>()
            .translate(move_vec * dt * 5.0);
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::with_user_event().build()?;

    let mut app = App::<Trig>::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;

    Ok(())
}
