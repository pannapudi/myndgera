// https://github.com/JuanDiegoMontoya/Frogfood/blob/main/src/techniques/Bloom.cpp

use std::{mem, sync::Arc};

use anyhow::Result;
use ash::vk;
use glam::{UVec2, uvec2};

use crate::{
    AppState, COLOR_SUBRESOURCE_MASK, ComputeHandle, Device, FrameGuard, ImageHandle,
    RenderContext, ScreenRelation, dispatch_optimal,
};

#[derive(Clone, Copy, Debug)]
pub struct BloomParams {
    pub target_image: ImageHandle,
    pub target_current_layout: vk::ImageLayout,
    pub strength: f32,
    pub width: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DownsamplePC {
    source_sampled_img_idx: u32,
    target_storage_img_idx: u32,
    source_dims: UVec2,
    target_dims: UVec2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct UpsamplePC {
    source_dims: UVec2,
    target_dims: UVec2,
    source_sampled_img_idx: u32,
    target_sampled_img_idx: u32,
    target_storage_img_idx: u32,
    width: f32,
    strength: f32,
    num_passes: u32,
    is_final_pass: u32,
}

pub struct Bloom {
    downsample_pass: ComputeHandle,
    downsample_low_pass: ComputeHandle,
    upsample_pass: ComputeHandle,
    miplevel_count: u32,
    accum_texture: ImageHandle,
    device: Arc<Device>,
}

impl Bloom {
    pub fn new(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        let miplevel_count = 6;
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(mem::size_of::<DownsamplePC>() as u32);
        let desc_layouts = [
            state.texture_arena.sampled_set_layout,
            state.texture_arena.storage_set_layout,
        ];
        let downsample_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_downsample.comp.glsl",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let downsample_low_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_downsample_low.comp.glsl",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(mem::size_of::<UpsamplePC>() as u32);
        let upsample_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_upsample.comp.glsl",
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
            .mip_levels(miplevel_count)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let accum_texture = state.texture_arena.push_image(
            texture_info,
            ScreenRelation::Half,
            &[],
            Some("Accumulation Texture"),
        )?;

        Ok(Self {
            downsample_pass,
            downsample_low_pass,
            upsample_pass,
            miplevel_count,
            accum_texture,
            device: ctx.device.clone(),
        })
    }

    pub fn apply(
        &self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &FrameGuard,
        params: BloomParams,
    ) {
        let _marker = self
            .device
            .create_scoped_marker(frame.command_buffer(), "Bloom Pass");
        let screen_extent = ctx.swapchain.extent;
        let source_image = params.target_image;
        let texture_arena = &mut state.texture_arena;

        ctx.device.image_transition(
            frame.command_buffer(),
            &texture_arena.get_image(source_image).inner,
            params.target_current_layout,
            vk::ImageLayout::GENERAL,
        );
        ctx.device.image_transition(
            frame.command_buffer(),
            &texture_arena.get_image(self.accum_texture).inner,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        let image_barriers = {
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .subresource_range(COLOR_SUBRESOURCE_MASK)
                .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            [
                image_barrier.image(texture_arena.get_image(params.target_image).inner),
                image_barrier.image(texture_arena.get_image(self.accum_texture).inner),
            ]
        };

        for i in 0..self.miplevel_count {
            ctx.device.pipeline_barrier(
                frame.command_buffer(),
                &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
            );

            let target_dims = uvec2(screen_extent.width, screen_extent.height) >> (i + 1);

            let (pipeline, source_lod, source_texture, source_dims, workgroup_size);
            if i == 0 {
                pipeline = self.downsample_low_pass;
                source_lod = 0;
                source_texture = params.target_image;
                source_dims = uvec2(screen_extent.width, screen_extent.height);
                workgroup_size = 16;
            } else {
                pipeline = self.downsample_pass;
                source_lod = i - 1;
                source_texture = self.accum_texture;
                source_dims = uvec2(screen_extent.width, screen_extent.height) >> i;
                workgroup_size = 8;
            }

            let downsample_pipeline = state.pipeline_arena.get_pipeline(pipeline);
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, downsample_pipeline);
            frame.bind_push_constants(
                downsample_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[DownsamplePC {
                    source_sampled_img_idx: texture_arena
                        .get_sampled_idx(source_texture, source_lod),
                    target_storage_img_idx: texture_arena.get_storage_idx(self.accum_texture, i),
                    source_dims,
                    target_dims,
                }],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                downsample_pipeline.layout,
                &[texture_arena.sampled_set, texture_arena.storage_set],
            );

            frame.dispatch(
                dispatch_optimal(target_dims.x, workgroup_size),
                dispatch_optimal(target_dims.y, workgroup_size),
                1,
            );
        }

        let upsample_pipeline = state.pipeline_arena.get_pipeline(self.upsample_pass);
        frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, upsample_pipeline);
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            upsample_pipeline.layout,
            &[texture_arena.sampled_set, texture_arena.storage_set],
        );

        for i in (0..self.miplevel_count).rev() {
            ctx.device.pipeline_barrier(
                frame.command_buffer(),
                &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
            );

            let source_dims = uvec2(screen_extent.width, screen_extent.height) >> (i + 1);
            let (target_lod, target_texture, target_dims);
            if i == 0 {
                target_lod = 0;
                target_texture = params.target_image;
                target_dims = uvec2(screen_extent.width, screen_extent.height);
            } else {
                target_lod = i - 1;
                target_texture = self.accum_texture;
                target_dims = uvec2(screen_extent.width, screen_extent.height) >> i;
            }

            frame.bind_push_constants(
                upsample_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[UpsamplePC {
                    source_sampled_img_idx: texture_arena.get_sampled_idx(self.accum_texture, i),
                    target_sampled_img_idx: texture_arena
                        .get_sampled_idx(target_texture, target_lod),
                    target_storage_img_idx: texture_arena
                        .get_storage_idx(target_texture, target_lod),
                    source_dims,
                    target_dims,
                    width: params.width,
                    strength: params.strength,
                    num_passes: self.miplevel_count,
                    is_final_pass: (i == 0) as u32,
                }],
            );
            frame.dispatch(
                dispatch_optimal(target_dims.x, 16),
                dispatch_optimal(target_dims.y, 16),
                1,
            );
        }
        ctx.device.image_transition(
            frame.command_buffer(),
            &texture_arena.get_image(source_image).inner,
            vk::ImageLayout::GENERAL,
            params.target_current_layout,
        );
    }
}
