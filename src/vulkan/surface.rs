use std::ops::Deref;

use anyhow::Result;
use ash::{khr, vk};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::instance::Instance;

pub struct Surface {
    pub loader: khr::surface::Instance,
    pub inner: vk::SurfaceKHR,
}

impl Deref for Surface {
    type Target = vk::SurfaceKHR;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceInfo {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl Surface {
    pub fn new(
        instance: &Instance,
        handle: &(impl HasDisplayHandle + HasWindowHandle),
    ) -> Result<Self> {
        let inner = unsafe {
            ash_window::create_surface(
                &instance.entry,
                instance,
                handle.display_handle()?.as_raw(),
                handle.window_handle()?.as_raw(),
                None,
            )?
        };

        let loader = khr::surface::Instance::new(&instance.entry, instance);

        Ok(Surface { inner, loader })
    }

    pub fn get_device_capabilities(
        &self,
        device: &vk::PhysicalDevice,
    ) -> vk::SurfaceCapabilitiesKHR {
        unsafe {
            self.loader
                .get_physical_device_surface_capabilities(*device, self.inner)
                .unwrap()
        }
    }

    pub fn get_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
    ) -> bool {
        unsafe {
            self.loader
                .get_physical_device_surface_support(
                    physical_device,
                    queue_family_index,
                    self.inner,
                )
                .unwrap()
        }
    }

    pub fn info(&self, device: &vk::PhysicalDevice) -> SurfaceInfo {
        let formats = unsafe {
            self.loader
                .get_physical_device_surface_formats(*device, self.inner)
                .unwrap()
        };

        let capabilities = unsafe {
            self.loader
                .get_physical_device_surface_capabilities(*device, self.inner)
                .unwrap()
        };

        let present_modes = unsafe {
            self.loader
                .get_physical_device_surface_present_modes(*device, self.inner)
                .unwrap()
        };

        SurfaceInfo {
            capabilities,
            formats,
            present_modes,
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.inner, None) };
    }
}
