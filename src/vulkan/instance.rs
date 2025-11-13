use std::{collections::HashSet, ffi::CStr};

use anyhow::Result;
use ash::{Entry, ext, vk};
use tracing::{Level, error};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use super::{device::Device, surface::Surface};

pub struct Instance {
    pub entry: ash::Entry,
    pub inner: ash::Instance,
    _dbg_loader: ext::debug_utils::Instance,
    _dbg_callbk: vk::DebugUtilsMessengerEXT,
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

        let mut required_extensions = vec![ext::debug_utils::NAME.as_ptr()];
        if let Some(handle) = display_handle {
            required_extensions.extend(ash_window::enumerate_required_extensions(
                handle.display_handle()?.as_raw(),
            )?);
        }
        let validation_layers = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation".as_ptr(),
        ];

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
        }

        let props = unsafe { entry.enumerate_instance_extension_properties(None) }?;
        let available_extensions = props
            .iter()
            .filter_map(|ext| ext.extension_name_as_c_str().ok())
            .collect::<HashSet<_>>();
        let extension_names = HashSet::from_iter(
            required_extensions
                .iter()
                .map(|&ext| unsafe { CStr::from_ptr(ext) }),
        );
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

        let mut debug_utils_info = {
            let message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
            let message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

            vk::DebugUtilsMessengerCreateInfoEXT::default()
                .pfn_user_callback(Some(vulkan_debug_utils_callback))
                .message_severity(message_severity)
                .message_type(message_type)
        };

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(create_flags)
            .enabled_layer_names(&validation_layers)
            .enabled_extension_names(&required_extensions)
            .push_next(&mut debug_utils_info);
        let inner = unsafe { entry.create_instance(&instance_info, None) }?;

        let dbg_loader = ext::debug_utils::Instance::new(&entry, &inner);
        let dbg_callbk =
            unsafe { dbg_loader.create_debug_utils_messenger(&debug_utils_info, None) }?;

        Ok(Self {
            _dbg_loader: dbg_loader,
            _dbg_callbk: dbg_callbk,
            entry,
            inner,
        })
    }

    pub fn get_format_properties(
        &self,
        &device: &vk::PhysicalDevice,
        format: vk::Format,
    ) -> vk::FormatProperties {
        unsafe { self.get_physical_device_format_properties(device, format) }
    }

    pub fn create_device_and_queues(&self) -> Result<(Device, vk::Queue)> {
        Device::create_with_queues(self)
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
            self._dbg_loader
                .destroy_debug_utils_messenger(self._dbg_callbk, None);
            self.inner.destroy_instance(None);
        }
    }
}

unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        && message_severity < vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
    {
        return vk::FALSE;
    }

    use vk::DebugUtilsMessageSeverityFlagsEXT as DF;
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "general",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "performance",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "validation",
        _ => "?",
    };

    let c_message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    let message = c_message.to_str();
    match (message_severity, message) {
        (DF::VERBOSE, Ok(msg)) => tracing::event!(Level::TRACE, "[vk {types}] {msg}"),
        (DF::WARNING, Ok(msg)) => tracing::event!(Level::WARN, "[vk {types}] {msg}"),
        (DF::ERROR, Ok(msg)) => tracing::event!(Level::ERROR, "[vk {types}] {msg}"),
        (DF::INFO | _, Ok(msg)) => tracing::event!(Level::INFO, "[vk {types}] {msg}"),
        (_, Err(_)) => tracing::event!(Level::TRACE, "[vk {types}] {c_message:?}"),
    }

    vk::FALSE
}
