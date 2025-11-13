use std::{
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    sync::Arc,
};

use ash::vk;
use gpu_allocator::vulkan::Allocation;

use crate::utils;

use super::Device;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct GpuBuffer<T: Copy, const N: usize = 1> {
    pub size: u32,
    pub data: [T; N],
}

impl<T: Copy, const N: usize> GpuBuffer<T, N> {
    pub const SIZE: usize = mem::size_of::<Self>();
}

pub struct Buffer {
    pub address: u64,
    pub size: u64,
    pub buffer: vk::Buffer,
    pub memory: ManuallyDrop<Allocation>,
    pub(super) device: Arc<Device>,
}

impl Buffer {
    pub fn map_memory(&mut self) -> Option<&mut [u8]> {
        self.memory.mapped_slice_mut()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.device.destroy_resource(self.buffer);
        let memory = unsafe { ManuallyDrop::take(&mut self.memory) };
        self.device.destroy_resource(memory);
    }
}

pub struct BufferTyped<T: 'static> {
    pub address: u64,
    pub buffer: vk::Buffer,
    pub memory: ManuallyDrop<Allocation>,
    pub(super) device: Arc<Device>,
    pub(super) _marker: PhantomData<*mut T>,
}

impl<T> BufferTyped<T> {
    pub fn map_memory(&mut self) -> Option<&mut T> {
        self.memory
            .mapped_slice_mut()
            .map(|slice| utils::from_bytes::<T>(slice))
    }
}

impl<T> Drop for BufferTyped<T> {
    fn drop(&mut self) {
        self.device.destroy_resource(self.buffer);
        let memory = unsafe { ManuallyDrop::take(&mut self.memory) };
        self.device.destroy_resource(memory);
    }
}
