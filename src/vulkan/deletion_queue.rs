use super::{Device, TimelineSemaphore};
use ash::vk;
use gpu_allocator::vulkan::Allocation;

#[derive(Debug)]
pub enum DeletableResource {
    DeviceMemory(vk::DeviceMemory),
    Allocation(Allocation),

    Swapchain(vk::SwapchainKHR),
    Surface(vk::SurfaceKHR),

    Fence(vk::Fence),
    Semaphore(vk::Semaphore),

    CommandBuffer(vk::CommandBuffer),
    ImageView(vk::ImageView),
    Image(vk::Image),
    Buffer(vk::Buffer),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
}

impl From<vk::Fence> for DeletableResource {
    fn from(resource: vk::Fence) -> Self {
        Self::Fence(resource)
    }
}

impl From<vk::DeviceMemory> for DeletableResource {
    fn from(resource: vk::DeviceMemory) -> Self {
        Self::DeviceMemory(resource)
    }
}

impl From<vk::SwapchainKHR> for DeletableResource {
    fn from(resource: vk::SwapchainKHR) -> Self {
        Self::Swapchain(resource)
    }
}

impl From<vk::SurfaceKHR> for DeletableResource {
    fn from(resource: vk::SurfaceKHR) -> Self {
        Self::Surface(resource)
    }
}

impl From<vk::Semaphore> for DeletableResource {
    fn from(resource: vk::Semaphore) -> Self {
        Self::Semaphore(resource)
    }
}

impl From<TimelineSemaphore> for DeletableResource {
    fn from(resource: TimelineSemaphore) -> Self {
        Self::Semaphore(resource.inner)
    }
}

impl From<vk::Image> for DeletableResource {
    fn from(resource: vk::Image) -> Self {
        Self::Image(resource)
    }
}

impl From<vk::ImageView> for DeletableResource {
    fn from(resource: vk::ImageView) -> Self {
        Self::ImageView(resource)
    }
}

impl From<vk::Buffer> for DeletableResource {
    fn from(resource: vk::Buffer) -> Self {
        Self::Buffer(resource)
    }
}

impl From<vk::Pipeline> for DeletableResource {
    fn from(resource: vk::Pipeline) -> Self {
        Self::Pipeline(resource)
    }
}

impl From<vk::PipelineLayout> for DeletableResource {
    fn from(resource: vk::PipelineLayout) -> Self {
        Self::PipelineLayout(resource)
    }
}

impl From<Allocation> for DeletableResource {
    fn from(resource: Allocation) -> Self {
        Self::Allocation(resource)
    }
}

impl From<vk::CommandBuffer> for DeletableResource {
    fn from(resource: vk::CommandBuffer) -> Self {
        Self::CommandBuffer(resource)
    }
}

#[derive(Debug)]
pub struct PendingDeletion {
    timeline_value: u64,
    resource: DeletableResource,
}

#[derive(Default)]
pub struct DeletionQueue {
    pub(super) pending_deletions: Vec<PendingDeletion>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            pending_deletions: vec![],
        }
    }

    pub fn queue_deletion_after(
        &mut self,
        resource: impl Into<DeletableResource>,
        timeline_value: u64,
    ) {
        self.pending_deletions.push(PendingDeletion {
            resource: resource.into(),
            timeline_value,
        });
    }

    pub fn destroy_ready(&mut self, device: &Device) {
        let current_timeline_value = unsafe {
            device
                .get_semaphore_counter_value(*device.timeline_semaphore)
                .unwrap()
        };

        self.pending_deletions.sort_by_key(|d| d.timeline_value);
        let partition_point = self
            .pending_deletions
            .partition_point(|d| d.timeline_value <= current_timeline_value);

        for PendingDeletion { resource, .. } in self.pending_deletions.drain(..partition_point) {
            unsafe {
                destroy_resource_immediate(device, resource);
            }
        }
    }

    pub unsafe fn destroy_all_immediate(&mut self, device: &Device) {
        self.pending_deletions.sort_by_key(|d| d.timeline_value);

        for PendingDeletion { resource, .. } in self.pending_deletions.drain(..) {
            unsafe { destroy_resource_immediate(device, resource) };
        }
    }
}

unsafe fn destroy_resource_immediate(device: &Device, resource: impl Into<DeletableResource>) {
    use DeletableResource::*;

    let resource = resource.into();

    unsafe {
        match resource {
            DeviceMemory(res) => device.free_memory(res, None),
            Allocation(res) => device.dealloc_memory(res),

            Swapchain(res) => device.swapchain_fns.destroy_swapchain(res, None),
            Surface(res) => device.surface_fns.destroy_surface(res, None),

            Fence(res) => device.destroy_fence(res, None),
            Semaphore(res) => device.destroy_semaphore(res, None),

            ImageView(res) => device.destroy_image_view(res, None),
            Image(res) => device.device.destroy_image(res, None),
            Buffer(res) => device.destroy_buffer(res, None),
            CommandBuffer(res) => device.free_command_buffers(&[res]),

            Pipeline(res) => device.destroy_pipeline(res, None),
            PipelineLayout(res) => device.destroy_pipeline_layout(res, None),
        }
    }
}
