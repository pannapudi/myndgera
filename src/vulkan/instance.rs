use std::{collections::HashSet, ffi::CStr};

use anyhow::Result;
use ash::{Entry, ext, khr, vk};
use tracing::{debug, error, info, warn};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{Device, Surface};

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut std::ffi::c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }.to_string_lossy();
    match flag {
        Flag::VERBOSE => debug!("{:?} - {}", typ, message),
        Flag::INFO => info!("{:?} - {}", typ, message),
        Flag::WARNING => warn!("{:?} - {}", typ, message),
        _ => error!("{:?} - {}", typ, message),
    }
    vk::FALSE
}

pub struct Instance {
    pub entry: ash::Entry,
    pub inner: ash::Instance,
    dbg_loader: ext::debug_utils::Instance,
    dbg_callbk: vk::DebugUtilsMessengerEXT,
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Instance {
    pub fn new(display_handle: Option<&impl HasDisplayHandle>) -> Result<Self> {
        let entry = unsafe { Entry::load() }?;
        let layers = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation".as_ptr(),
        ];
        let mut extensions = vec![
            ext::debug_utils::NAME.as_ptr(),
            khr::get_physical_device_properties2::NAME.as_ptr(),
        ];
        if let Some(handle) = display_handle {
            extensions.extend(ash_window::enumerate_required_extensions(
                handle.display_handle()?.as_raw(),
            )?);
        }

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extensions.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let props = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let available_extensions = props
            .iter()
            .filter_map(|ext| ext.extension_name_as_c_str().ok())
            .collect::<HashSet<_>>();
        let extension_names =
            HashSet::from_iter(extensions.iter().map(|&ext| unsafe { CStr::from_ptr(ext) }));
        let mut missing = extension_names.difference(&available_extensions).peekable();
        if missing.peek().is_some() {
            error!("Missing instance extenstions:");
            missing.for_each(|s| println!("\t{}", s.to_string_lossy()));
        }

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let appinfo = vk::ApplicationInfo::default()
            .application_name(c"Modern Vulkan")
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(create_flags)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let inner = unsafe { entry.create_instance(&instance_info, None) }?;

        let dbg_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));
        let dbg_loader = ext::debug_utils::Instance::new(&entry, &inner);
        let dbg_callbk = unsafe { dbg_loader.create_debug_utils_messenger(&dbg_info, None)? };

        Ok(Self {
            dbg_loader,
            dbg_callbk,
            entry,
            inner,
        })
    }

    pub fn create_device_and_queues(&self, surface: &Surface) -> Result<(Device, vk::Queue)> {
        Device::create_with_queues(self, surface)
    }

    pub fn create_surface(
        &self,
        handle: &(impl HasDisplayHandle + HasWindowHandle),
    ) -> Result<Surface> {
        Surface::new(self, handle)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.dbg_loader
                .destroy_debug_utils_messenger(self.dbg_callbk, None);
            self.inner.destroy_instance(None);
        }
    }
}
