mod buffers;
mod device;
mod frame;
mod instance;
mod pipeline_arena;
mod staging;
mod surface;
mod swapchain;
mod texture_arena;
mod view_target;

use ash::vk;

pub use buffers::*;
pub use device::*;
pub use frame::*;
pub use instance::Instance;
pub use pipeline_arena::*;
pub use staging::*;
pub use surface::Surface;
pub use swapchain::*;
pub use texture_arena::*;
pub use view_target::*;

pub const BASE_IMAGE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

pub struct TimelineSemaphore {
    inner: vk::Semaphore,
    value: std::sync::atomic::AtomicU64,
}

impl TimelineSemaphore {
    pub fn new(device: &ash::Device, initial_value: Option<u64>) -> ash::prelude::VkResult<Self> {
        let initial_value = initial_value.unwrap_or(0);
        let mut semaphore_type = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type);
        let inner = unsafe { device.create_semaphore(&semaphore_info, None) }?;

        Ok(Self {
            inner,
            value: std::sync::atomic::AtomicU64::new(initial_value),
        })
    }

    pub fn advance(&self, to: u64) -> (u64, u64) {
        let wait_value = self.value();
        let signal_value = wait_value + to;
        self.value
            .fetch_add(to, std::sync::atomic::Ordering::Relaxed);
        (wait_value, signal_value)
    }

    pub fn value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl std::ops::Deref for TimelineSemaphore {
    type Target = vk::Semaphore;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
