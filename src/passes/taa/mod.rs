use std::mem;

use anyhow::Result;
use ash::vk;
use glam::{Vec2, vec2};
use rand::SeedableRng;
use rand::{rngs::StdRng, seq::SliceRandom};

use crate::{
    AppState, COLOR_SUBRESOURCE_MASK, ComputeHandle, FrameGuard, ImageHandle, RenderContext,
    ScreenRelation, ViewTarget, dispatch_optimal,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ReprojectPC {
    depth_image: u32,
    motion_image: u32,
    camera_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct TaaPC {
    src_image: u32,
    dst_image: u32,
    history_image: u32,
    motion_image: u32,
}

pub struct TaaParams<'a> {
    pub view_target: &'a ViewTarget,
    pub depth_image: ImageHandle,
}

#[inline]
fn radical_inverse(mut n: u32, base: u32) -> f32 {
    let mut val = 0.0f32;
    let inv_base = 1.0f32 / base as f32;
    let mut inv_bi = inv_base;

    while n > 0 {
        let d_i = n % base;
        val += d_i as f32 * inv_bi;
        n = (n as f32 * inv_base) as u32;
        inv_bi *= inv_base;
    }

    val
}

pub struct Taa {
    history_image: ImageHandle,
    motion_image: ImageHandle,
    reproject_pipeline: ComputeHandle,
    taa_pipeline: ComputeHandle,

    pub jitter_samples: Vec<Vec2>,
}

impl Taa {
    pub fn new(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(mem::size_of::<ReprojectPC>() as u32);
        let desc_layouts = [
            state.texture_arena.sampled_set_layout,
            state.texture_arena.storage_set_layout,
        ];
        let reproject_pipeline = state.pipeline_arena.create_compute_pipeline(
            "src/passes/taa/reproject.comp.glsl",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(mem::size_of::<TaaPC>() as u32);
        let taa_pipeline = state.pipeline_arena.create_compute_pipeline(
            "src/passes/taa/taa.comp.glsl",
            &[push_constant_range],
            &desc_layouts,
        )?;

        let texture_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: ctx.swapchain.extent.width,
                height: ctx.swapchain.extent.height,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B10G11R11_UFLOAT_PACK32)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let motion_image = state.texture_arena.push_image(
            texture_info,
            ScreenRelation::Identity,
            &[],
            Some("Motion Image"),
        )?;
        let history_image = state.texture_arena.push_image(
            texture_info,
            ScreenRelation::Identity,
            &[],
            Some("History Image"),
        )?;

        let n = 16;
        let jitter_samples = (0..n)
            .map(|i| {
                Vec2::new(
                    radical_inverse(i % n + 1, 2) * 2. - 1.,
                    radical_inverse(i % n + 1, 3) * 2. - 1.,
                )
            })
            .collect();

        Ok(Self {
            history_image,
            motion_image,
            reproject_pipeline,
            taa_pipeline,

            jitter_samples,
        })
    }

    pub fn get_jitter(&mut self, frame_idx: u32, width: u32, height: u32) -> Vec2 {
        if frame_idx.is_multiple_of(self.jitter_samples.len() as u32) {
            let mut rng = StdRng::seed_from_u64(frame_idx as u64);

            let prev_sample = self.jitter_samples.last().copied();
            loop {
                self.jitter_samples.shuffle(&mut rng);
                if self.jitter_samples.first().copied() != prev_sample {
                    break;
                }
            }
        }

        self.jitter_samples[frame_idx as usize % self.jitter_samples.len()]
            / vec2(width as f32, height as f32)
    }

    pub fn apply(
        &self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &FrameGuard,
        params: TaaParams,
    ) {
        let _marker = ctx
            .device
            .create_scoped_marker(frame.command_buffer(), "Taa Pass");
        let texture_arena = &mut state.texture_arena;
        let postprocess_write = params.view_target.post_process_write();

        let barrier = vk::ImageMemoryBarrier2::default()
            .subresource_range(COLOR_SUBRESOURCE_MASK)
            .image(texture_arena.get_image(*postprocess_write.source).inner)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
        ctx.device.pipeline_barrier(
            frame.command_buffer(),
            &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
        );

        {
            let reproject_push_constant = ReprojectPC {
                motion_image: texture_arena.get_storage_idx(self.motion_image, 0),
                depth_image: texture_arena.get_storage_idx(params.depth_image, 0),
                camera_buffer: state.camera_uniform_gpu.address,
            };

            let pipeline = state.pipeline_arena.get_pipeline(self.reproject_pipeline);
            frame.bind_push_constants(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[reproject_push_constant],
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

        let barrier = vk::ImageMemoryBarrier2::default()
            .subresource_range(COLOR_SUBRESOURCE_MASK)
            .image(texture_arena.get_image(self.motion_image).inner)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
        ctx.device.pipeline_barrier(
            frame.command_buffer(),
            &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
        );

        {
            let taa_push_constant = TaaPC {
                src_image: texture_arena.get_storage_idx(*postprocess_write.source, 0),
                dst_image: texture_arena.get_storage_idx(*postprocess_write.destination, 0),
                motion_image: texture_arena.get_storage_idx(self.motion_image, 0),
                history_image: texture_arena.get_storage_idx(self.history_image, 0),
            };

            let pipeline = state.pipeline_arena.get_pipeline(self.taa_pipeline);
            frame.bind_push_constants(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[taa_push_constant],
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

        let barrier = barrier
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
        ctx.device.pipeline_barrier(
            frame.command_buffer(),
            &vk::DependencyInfo::default().image_memory_barriers(&[barrier]),
        );
    }
}
