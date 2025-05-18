use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use super::Device;

pub struct Frame {
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub present_finished: vk::Fence,
}

impl Frame {
    pub fn new(device: &Device) -> VkResult<Self> {
        let image_available_semaphore = device.create_semaphore()?;
        let render_finished_semaphore = device.create_semaphore()?;
        let present_finished = device.create_fence(vk::FenceCreateFlags::SIGNALED)?;
        Ok(Self {
            image_available_semaphore,
            render_finished_semaphore,
            present_finished,
        })
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_fence(self.present_finished, None);
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
        }
    }
}

pub struct FrameGuard {
    pub frame: Frame,
    pub command_buffer: vk::CommandBuffer,
    pub extent: vk::Extent2D,
    pub image_idx: usize,
    pub device: Arc<Device>,
}

impl FrameGuard {
    pub fn command_buffer(&self) -> &vk::CommandBuffer {
        &self.command_buffer
    }

    pub fn begin_rendering(
        &self,
        image: &vk::Image,
        view: &vk::ImageView,
        load_op: vk::AttachmentLoadOp,
        color: [f32; 4],
    ) {
        self.device.image_transition(
            self.command_buffer(),
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(*view)
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
            .begin_rendering(&self.command_buffer, &rendering_info);

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
    }
}
