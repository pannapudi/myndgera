use std::{slice, sync::Arc};

use anyhow::{Context, Result, ensure};
use arrayvec::ArrayVec;
use ash::{
    prelude::VkResult,
    vk::{self, Extent2D},
};
use tracing::debug;
use winit::window::Window;

use crate::COLOR_SUBRESOURCE_MASK;

use super::{Device, ImageDimensions, Instance, Surface};

const MAX_IMAGE_CAP: usize = 16;

pub struct Frame {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    present_finished: vk::Fence,

    pub prev_submit_timeline_value: u64,
}

impl Frame {
    fn new(device: &Device, idx: u64) -> VkResult<Self> {
        let image_available_semaphore = device.create_semaphore()?;
        device.name_object(
            image_available_semaphore,
            &format!("Image Available Semaphore {idx}"),
        );
        let render_finished_semaphore = device.create_semaphore()?;
        device.name_object(
            render_finished_semaphore,
            &format!("Render Finished Semaphore {idx}"),
        );
        let present_finished = device.create_fence(vk::FenceCreateFlags::SIGNALED)?;
        device.name_object(present_finished, &format!("Present Finished Fence {idx}"));

        Ok(Frame {
            image_available_semaphore,
            render_finished_semaphore,
            present_finished,
            prev_submit_timeline_value: idx,
        })
    }
}

pub struct Swapchain {
    pub surface: Surface,
    pub format: vk::Format,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,

    pub frames: ArrayVec<Frame, MAX_IMAGE_CAP>,
    next_sync_idx: usize,
    guts: InnerGuts,
    device: Arc<Device>,
}

impl Swapchain {
    pub fn current_frame(&self) -> &Frame {
        &self.frames[self.next_sync_idx]
    }

    pub fn current_image(&self) -> &vk::Image {
        &self.guts.images[self.next_sync_idx]
    }

    pub fn image_dimensions(&self) -> ImageDimensions {
        let Extent2D { width, height } = self.extent();
        let memory_reqs = unsafe {
            self.device
                .get_image_memory_requirements(self.guts.images[0])
        };
        ImageDimensions::new(width as _, height as _, memory_reqs.alignment)
    }

    pub fn new(device: &Arc<Device>, instance: &Instance, window: &Window) -> Result<Self> {
        let surface = Surface::new(instance, &window)?;
        let surface_info = surface.info(&device.physical_device);

        let extent = match surface_info.capabilities.current_extent {
            vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            } => {
                let (width, height) = window.inner_size().into();
                vk::Extent2D { width, height }
            }

            current => current,
        };
        debug!("Swapchain extent: {:?}", extent);

        let present_mode = surface_info
            .present_modes
            .into_iter()
            .max_by_key(|&mode| match mode {
                vk::PresentModeKHR::FIFO => 1,
                vk::PresentModeKHR::FIFO_RELAXED => 2,
                _ => 0,
            })
            .context("Selecting supported present mode")?;
        debug!("Swapchain present mode: {:?}", present_mode);

        let format = surface_info
            .formats
            .into_iter()
            .filter(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .filter(|format| {
                let props = instance.get_format_properties(&device.physical_device, format.format);
                props.optimal_tiling_features.contains(
                    vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::SAMPLED_IMAGE,
                )
            })
            .map(|format| format.format)
            .max_by_key(|&format| match format {
                vk::Format::R8G8B8A8_SRGB => 15,
                vk::Format::B8G8R8A8_SRGB => 14,
                vk::Format::A8B8G8R8_SRGB_PACK32 => 13,

                vk::Format::R8G8B8A8_UNORM => 5,
                vk::Format::B8G8R8A8_UNORM => 4,
                vk::Format::A8B8G8R8_UNORM_PACK32 => 3,
                _ => 0,
            })
            .context("Selecting supported swapchain format")?;

        let num_images = (surface_info.capabilities.min_image_count + 1).min(MAX_IMAGE_CAP as u32);
        debug!("Swapchain image count: {:?}", num_images);

        ensure!(
            surface_info
                .capabilities
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        );

        let guts = InnerGuts::new(
            device,
            &surface,
            format,
            present_mode,
            extent,
            num_images,
            None,
        )?;

        let frames = (0..guts.images.len())
            .map(|i| Frame::new(device, i as u64))
            .collect::<VkResult<ArrayVec<_, MAX_IMAGE_CAP>>>()?;

        Ok(Self {
            surface,
            format,
            present_mode,
            extent,

            guts,
            frames,
            next_sync_idx: 0,
            device: device.clone(),
        })
    }

    pub fn resize(&mut self, new_extent: vk::Extent2D) -> VkResult<()> {
        if self.extent == new_extent {
            return Ok(());
        }

        let (min, max) = {
            let caps = self
                .surface
                .get_device_capabilities(&self.device.physical_device);
            (caps.min_image_extent, caps.max_image_extent)
        };
        if (new_extent.width < min.width || new_extent.width > max.width || new_extent.width == 0)
            || (new_extent.height < min.height
                || new_extent.height > max.height
                || new_extent.height == 0)
        {
            self.extent = vk::Extent2D {
                width: 0,
                height: 0,
            };
            return Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
        }

        let new_swapchain = InnerGuts::new(
            &self.device,
            &self.surface,
            self.format,
            self.present_mode,
            new_extent,
            self.guts.images.len() as u32,
            Some(&self.guts),
        )?;

        self.extent = new_extent;
        let old_swapchain = std::mem::replace(&mut self.guts, new_swapchain);

        for view in old_swapchain.views {
            self.device.destroy_resource(view);
        }
        self.device.destroy_resource(old_swapchain.swapchain);

        Ok(())
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.guts.images
    }

    pub fn views(&self) -> &[vk::ImageView] {
        &self.guts.views
    }

    pub fn get_image(&self, idx: usize) -> vk::Image {
        self.guts.images[idx]
    }

    pub fn get_view(&self, idx: usize) -> vk::ImageView {
        self.guts.views[idx]
    }

    pub fn start_frame(&mut self) -> VkResult<FrameGuard> {
        let sync_idx = self.next_sync_idx;
        self.next_sync_idx = (self.next_sync_idx + 1) % self.frames.len();

        let frame = &self.frames[sync_idx];

        let image_idx = match self
            .device
            .acquire_next_image(&self.guts.swapchain, &frame.image_available_semaphore)
        {
            Ok((idx, false)) => idx as usize,
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
            }
            Err(e) => return Err(e),
        };

        self.device
            .wait_for_fences(&[frame.present_finished], true, u64::MAX)?;
        self.device.reset_fences(&[frame.present_finished])?;

        let cbuff = self.device.allocate_command_buffer()?;
        self.device
            .begin_command_buffer(&cbuff, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

        Ok(FrameGuard {
            cbuff,
            sync_idx,
            image_idx,
            extent: self.extent,
            device: self.device.clone(),
            image: None,
            view: None,
        })
    }

    pub fn submit_frame(&mut self, frame_guard: FrameGuard) -> VkResult<()> {
        let frame = &mut self.frames[frame_guard.sync_idx];
        let image_idx = frame_guard.image_idx;

        self.device.end_command_buffer(&frame_guard.cbuff)?;

        let timeline_semaphore = &self.device.timeline_semaphore;
        let (_wait_value, signal_value) = timeline_semaphore.advance(1);
        frame.prev_submit_timeline_value = signal_value;

        let wait_semaphores_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signal_semaphores_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.render_finished_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(**timeline_semaphore)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .value(signal_value),
        ];

        let cbuf_info = vk::CommandBufferSubmitInfo::default().command_buffer(frame_guard.cbuff);
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphores_info)
            .signal_semaphore_infos(&signal_semaphores_info)
            .command_buffer_infos(slice::from_ref(&cbuf_info));
        self.device
            .queue_submit(&self.device.queue, &[submit_info], None)?;

        self.device.destroy_resource(frame_guard.cbuff);

        match self.device.queue_present(
            &self.device.queue,
            &self.guts.swapchain,
            &frame.render_finished_semaphore,
            &frame.present_finished,
            image_idx,
        ) {
            Ok(false) => Ok(()),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
            }
            Err(e) => Err(e),
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.frames.iter_mut().for_each(|f| {
                self.device
                    .destroy_semaphore(f.image_available_semaphore, None);
                self.device
                    .destroy_semaphore(f.render_finished_semaphore, None);
                self.device.destroy_fence(f.present_finished, None);
            });
            self.guts.views.iter().for_each(|&view| {
                self.device.destroy_image_view(view, None);
            });
            self.device
                .swapchain_fns
                .destroy_swapchain(self.guts.swapchain, None);
        }
    }
}

struct InnerGuts {
    swapchain: vk::SwapchainKHR,
    images: ArrayVec<vk::Image, MAX_IMAGE_CAP>,
    views: ArrayVec<vk::ImageView, MAX_IMAGE_CAP>,
}

impl InnerGuts {
    fn new(
        device: &Device,
        surface: &Surface,
        format: vk::Format,
        present_mode: vk::PresentModeKHR,
        extent: vk::Extent2D,
        num_images: u32,
        old_swapchain: Option<&Self>,
    ) -> VkResult<Self> {
        let format_srgb = match format {
            vk::Format::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_UNORM => vk::Format::B8G8R8A8_SRGB,
            vk::Format::A8B8G8R8_UNORM_PACK32 => vk::Format::A8B8G8R8_SRGB_PACK32,
            x => x,
        };

        let formats = [format, format_srgb];
        let mut format_list_info = vk::ImageFormatListCreateInfo::default().view_formats(&formats);

        let mut swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .min_image_count(num_images)
            .image_format(format)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        if let Some(old_swapchain) = old_swapchain {
            swapchain_info.old_swapchain = old_swapchain.swapchain;
        }

        if format_srgb != format {
            swapchain_info = swapchain_info
                .flags(vk::SwapchainCreateFlagsKHR::MUTABLE_FORMAT)
                .push_next(&mut format_list_info);
        }

        let inner = unsafe { device.swapchain_fns.create_swapchain(&swapchain_info, None) }?;

        let images = {
            let images = unsafe { device.swapchain_fns.get_swapchain_images(inner)? };
            ArrayVec::from_iter(images)
        };
        let views = images
            .iter()
            .map(|img| device.create_2d_view(img, format, 0))
            .collect::<VkResult<ArrayVec<_, MAX_IMAGE_CAP>>>()?;

        for (i, (&image, &view)) in std::iter::zip(&images, &views).enumerate() {
            device.name_object(image, &format!("Swapchain Image {i}"));
            device.name_object(view, &format!("Swapchain View {i}"));
        }

        Ok(InnerGuts {
            swapchain: inner,
            images,
            views,
        })
    }
}

pub struct FrameGuard {
    sync_idx: usize,
    pub cbuff: vk::CommandBuffer,
    image: Option<vk::Image>,
    view: Option<vk::ImageView>,
    pub image_idx: usize,

    pub extent: vk::Extent2D,
    pub device: Arc<Device>,
}

impl FrameGuard {
    pub fn command_buffer(&self) -> &vk::CommandBuffer {
        &self.cbuff
    }

    pub fn begin_rendering(
        &mut self,
        &image: &vk::Image,
        &view: &vk::ImageView,
        load_op: vk::AttachmentLoadOp,
        color: [f32; 4],
    ) {
        self.image = Some(image);
        self.view = Some(view);
        let image_barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )
            .subresource_range(COLOR_SUBRESOURCE_MASK);
        self.device.pipeline_barrier(
            self.command_buffer(),
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&image_barrier)),
        );

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_image_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.extent.into())
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment));
        self.device
            .begin_rendering(self.command_buffer(), &rendering_info);

        let viewport = vk::Viewport {
            x: 0.0,
            y: self.extent.height as f32,
            width: self.extent.width as f32,
            height: -(self.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.set_viewports(&[viewport]);
        self.set_scissors(&[vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }]);
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.device.draw(
            self.command_buffer(),
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.device.draw_indexed(
            self.command_buffer(),
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64) {
        self.device
            .bind_index_buffer(self.command_buffer(), buffer, offset);
    }

    pub fn bind_vertex_buffer(&self, buffer: vk::Buffer) {
        self.device
            .bind_vertex_buffer(self.command_buffer(), buffer);
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        self.device.bind_descriptor_sets(
            self.command_buffer(),
            bind_point,
            pipeline_layout,
            descriptor_sets,
        );
    }

    pub fn bind_push_constants<T>(
        &self,
        pipeline_layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        data: &[T],
    ) {
        self.device
            .bind_push_constants(self.command_buffer(), pipeline_layout, stages, data);
    }

    pub fn set_viewports(&self, viewports: &[vk::Viewport]) {
        self.device.set_viewports(self.command_buffer(), viewports)
    }

    pub fn set_scissors(&self, viewports: &[vk::Rect2D]) {
        self.device.set_scissors(self.command_buffer(), viewports)
    }

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: &vk::Pipeline) {
        self.device
            .bind_pipeline(self.command_buffer(), bind_point, pipeline)
    }

    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.device.dispatch(self.command_buffer(), x, y, z)
    }

    pub fn end_rendering(&self) {
        self.device.end_rendering(self.command_buffer());

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .image(self.image.unwrap())
            .old_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .subresource_range(COLOR_SUBRESOURCE_MASK);
        self.device.pipeline_barrier(
            self.command_buffer(),
            &vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&image_barrier)),
        );
    }
}
