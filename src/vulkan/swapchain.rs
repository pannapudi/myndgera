use std::{slice, sync::Arc, time::Duration};

use ash::{
    khr,
    prelude::VkResult,
    vk::{self, Extent2D},
};
use tracing::debug;

use super::{BASE_IMAGE_RANGE, Device, Frame, FrameGuard, ImageDimensions, Surface};

pub struct Swapchain {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
    pub frames: Vec<Option<Frame>>,
    pub views: Vec<vk::ImageView>,
    pub images: Vec<vk::Image>,
    pub loader: khr::swapchain::Device,
    pub inner: vk::SwapchainKHR,
    reclaimed_semaphores: Vec<vk::Semaphore>,
    current_frame: usize,
    device: Arc<Device>,
}

impl Swapchain {
    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn image_dimensions(&self) -> ImageDimensions {
        let Extent2D { width, height } = self.extent();
        let memory_reqs = unsafe { self.device.get_image_memory_requirements(self.images[0]) };
        ImageDimensions::new(width as _, height as _, memory_reqs.alignment)
    }

    pub fn new(
        device: &Arc<Device>,
        surface: &Surface,
        swapchain_loader: khr::swapchain::Device,
        width: u32,
        height: u32,
    ) -> VkResult<Self> {
        let surface_info = surface.info(device);

        let format = surface_info
            .formats
            .iter()
            .find(|format| format.format == vk::Format::B8G8R8A8_UNORM)
            .unwrap_or(&surface_info.formats[0]);
        debug!("Swapchain format: {:?}", format);

        // Swapchain present mode
        let present_mode = surface_info
            .present_modes
            .iter()
            .cloned()
            .find(|&present_mode| present_mode == vk::PresentModeKHR::FIFO)
            .unwrap_or(surface_info.present_modes[0]);
        debug!("Swapchain present mode: {:?}", present_mode);

        let capabilities = surface_info.capabilities;

        // Swapchain extent
        let extent = {
            let max = capabilities.max_image_extent;
            let min = capabilities.min_image_extent;
            let width = width.min(max.width).max(min.width);
            let height = height.min(max.height).max(min.height);
            vk::Extent2D { width, height }
        };
        debug!("Swapchain extent: {:?}", extent);

        // Swapchain image count
        let image_count = capabilities
            .max_image_count
            .min(3)
            .max(capabilities.min_image_count);
        debug!("Swapchain image count: {:?}", image_count);

        let queue_family_index = [device.main_queue_family_idx];

        // Swapchain
        assert!(
            capabilities
                .supported_composite_alpha
                .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        );
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.inner)
            .image_format(format.format)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_extent(extent)
            .image_color_space(format.color_space)
            .min_image_count(image_count)
            .image_array_layers(1)
            .queue_family_indices(&queue_family_index)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        // Swapchain images and image views
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        images
            .iter()
            .enumerate()
            .for_each(|(i, &image)| device.name_object(image, &format!("Swapchain Image {i}")));

        let views = images
            .iter()
            .map(|img| device.create_2d_view(img, format.format, 0))
            .collect::<VkResult<Vec<_>>>()?;
        views
            .iter()
            .enumerate()
            .for_each(|(i, &view)| device.name_object(view, &format!("Swapchain View {i}")));

        let frames = (0..images.len())
            .map(|_| Some(Frame::new(device)).transpose())
            .collect::<VkResult<Vec<Option<Frame>>>>()?;

        Ok(Self {
            device: device.clone(),
            loader: swapchain_loader,
            inner: swapchain,
            present_mode,
            extent,
            format: *format,
            frames,
            images,
            views,
            current_frame: 0,
            reclaimed_semaphores: vec![],
        })
    }

    pub fn recreate(&mut self, surface: &Surface, width: u32, height: u32) -> VkResult<()> {
        debug!("Surface has been recreated: {{ width: {width}, height: {height} }}");

        let info = surface.info(&self.device);
        let capabilities = info.capabilities;

        for view in self.views.iter() {
            unsafe { self.device.destroy_image_view(*view, None) };
        }
        let old_swapchain = self.inner;

        let queue_family_index = [self.device.main_queue_family_idx];

        self.extent = {
            let max = capabilities.max_image_extent;
            let min = capabilities.min_image_extent;
            let width = width.min(max.width).max(min.width);
            let height = height.min(max.height).max(min.height);
            vk::Extent2D { width, height }
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .old_swapchain(old_swapchain)
            .image_format(self.format.format)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_extent(self.extent)
            .image_color_space(self.format.color_space)
            .min_image_count(self.images.len() as u32)
            .image_array_layers(1)
            .queue_family_indices(&queue_family_index)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true);
        self.inner = unsafe { self.loader.create_swapchain(&swapchain_create_info, None)? };

        unsafe { self.loader.destroy_swapchain(old_swapchain, None) };

        self.images = unsafe { self.loader.get_swapchain_images(self.inner)? };
        self.images.iter().enumerate().for_each(|(i, &image)| {
            self.device
                .name_object(image, &format!("Swapchain Image {i}"))
        });

        self.views = self
            .images
            .iter()
            .map(|img| self.device.create_2d_view(img, self.format.format, 0))
            .collect::<VkResult<Vec<_>>>()?;
        self.views.iter().enumerate().for_each(|(i, &view)| {
            self.device
                .name_object(view, &format!("Swapchain View {i}"))
        });

        Ok(())
    }

    pub fn get_current_frame(&self) -> Option<&Frame> {
        self.frames.get(self.current_frame).and_then(Option::as_ref)
    }
    pub fn get_current_frame_mut(&mut self) -> Option<&mut Frame> {
        self.frames
            .get_mut(self.current_frame)
            .and_then(Option::as_mut)
    }
    pub fn get_current_image(&self) -> &vk::Image {
        &self.images[self.current_frame]
    }
    pub fn get_current_view(&self) -> &vk::ImageView {
        &self.views[self.current_frame]
    }

    pub fn acquire_next_image(&mut self) -> VkResult<FrameGuard> {
        let acquire_semaphore = self
            .reclaimed_semaphores
            .pop()
            .map_or_else(|| self.device.create_semaphore(), Ok)?;
        let one_second = Duration::from_secs(1).as_nanos() as u64;

        let image_idx = match unsafe {
            self.loader.acquire_next_image(
                self.inner,
                one_second,
                acquire_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((idx, false)) => idx as usize,
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.reclaimed_semaphores.push(acquire_semaphore);
                return VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
            }
            Err(e) => return Err(e),
        };
        self.current_frame = image_idx;
        let Some(mut frame) = self.frames[image_idx].take() else {
            return Err(vk::Result::ERROR_UNKNOWN);
        };
        self.reclaimed_semaphores.push(std::mem::replace(
            &mut frame.image_available_semaphore,
            acquire_semaphore,
        ));

        self.device
            .wait_for_fences(&[frame.present_finished], true, one_second)?;
        unsafe { self.device.reset_fences(&[frame.present_finished])? };

        let command_buffer = self.device.start_command_buffer()?;

        self.device.image_transition(
            &command_buffer,
            &self.images[image_idx],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        Ok(FrameGuard {
            frame,
            command_buffer,
            extent: self.extent,
            image_idx,
            device: self.device.clone(),
        })
    }

    pub fn submit_image(&mut self, frame_guard: FrameGuard) -> VkResult<()> {
        let frame = frame_guard.frame;
        let command_buffer = &frame_guard.command_buffer;

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.images[frame_guard.image_idx])
            .subresource_range(BASE_IMAGE_RANGE);
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));
        self.device
            .pipeline_barrier(command_buffer, &dependency_info);

        self.device.end_command_buffer(command_buffer)?;

        let wait_semaphores = [frame.image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(slice::from_ref(command_buffer))
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device
                .queue_submit(self.device.queue, &[submit_info], frame.present_finished)?
        };

        let image_indices = [frame_guard.image_idx as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(slice::from_ref(&self.inner))
            .image_indices(&image_indices);

        self.frames[self.current_frame] = Some(frame);

        match unsafe { self.loader.queue_present(self.device.queue, &present_info) } {
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
            self.frames
                .iter_mut()
                .filter_map(|f| f.as_mut())
                .for_each(|f| f.destroy(&self.device));
            self.views.iter().for_each(|&view| {
                self.device.destroy_image_view(view, None);
            });
            self.reclaimed_semaphores.iter().for_each(|&sema| {
                self.device.destroy_semaphore(sema, None);
            });
            self.loader.destroy_swapchain(self.inner, None);
        }
    }
}
