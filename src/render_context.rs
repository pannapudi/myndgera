use crate::vulkan::{self, Device, Instance, Swapchain};
use anyhow::Result;
use std::sync::Arc;
use winit::window::Window;

pub struct RenderContext {
    pub(crate) is_swapchain_dirty: bool,
    pub swapchain: Swapchain,

    pub device: Arc<Device>,
    _instance: Instance,
    pub window: Window,
}

impl RenderContext {
    pub fn new(window: Window) -> Result<Self> {
        let instance = Instance::new(Some(&window))?;

        let (device, _transfer_queue) = instance.create_device_and_queues()?;
        let device = Arc::new(device);

        let swapchain = vulkan::Swapchain::new(&device, &instance, &window)?;

        Ok(Self {
            window,
            _instance: instance,
            device,
            swapchain,
            is_swapchain_dirty: false,
        })
    }
}
