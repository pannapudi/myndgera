use std::sync::Arc;

use anyhow::{Context, Result};
use ash::{prelude::VkResult, vk};
use gpu_allocator::MemoryLocation;

use super::{Buffer, Device};

const INITIAL_BUFFER_SIZE: u64 = 1024;

struct PendingWrite {
    region: vk::BufferCopy2<'static>,
    buffer: vk::Buffer,
}

pub struct StagingWrite {
    buffer: Buffer,
    intermediate_data: Vec<u8>,
    pending_writes: Vec<PendingWrite>,
    device: Arc<Device>,
}

impl StagingWrite {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let buffer = device.create_buffer(
            INITIAL_BUFFER_SIZE,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        device.name_object(*buffer, "Staging Buffer");
        Ok(Self {
            buffer,
            intermediate_data: vec![],
            pending_writes: vec![],
            device: device.clone(),
        })
    }

    pub fn write_buffer(&mut self, buffer: vk::Buffer, data: &[u8]) {
        let offset = self.intermediate_data.len();
        self.pending_writes.push(PendingWrite {
            buffer,
            region: vk::BufferCopy2::default()
                .src_offset(offset as u64)
                .dst_offset(0)
                .size(data.len() as u64),
        });
        self.intermediate_data.extend(data);
    }

    pub fn reserve_storage(&mut self) -> VkResult<Option<Buffer>> {
        let offset = self.intermediate_data.len();
        if offset < self.buffer.size as usize {
            return Ok(None);
        }

        let max_buffer_size = self
            .device
            .device_properties
            .limits
            .max_storage_buffer_range;
        let new_size = offset
            .checked_next_power_of_two()
            .unwrap_or(offset)
            .min(max_buffer_size as usize);
        let old_buffer = std::mem::replace(
            &mut self.buffer,
            self.device.create_buffer(
                new_size as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )?,
        );
        self.device.name_object(*self.buffer, "Staging Buffer");

        Ok(Some(old_buffer))
    }

    pub fn consume_pending_writes(&mut self, &cbuff: &vk::CommandBuffer) -> Result<Option<Buffer>> {
        if self.pending_writes.is_empty() {
            return Ok(None);
        }
        let old_buffer = self.reserve_storage()?;
        let mapped = self.buffer.map_memory().context("Failed to map memory")?;
        mapped[..self.intermediate_data.len()].copy_from_slice(&self.intermediate_data);
        for write in self.pending_writes.drain(..) {
            let copy_info = vk::CopyBufferInfo2::default()
                .src_buffer(self.buffer.buffer)
                .dst_buffer(write.buffer)
                .regions(std::slice::from_ref(&write.region));
            unsafe { self.device.cmd_copy_buffer2(cbuff, &copy_info) };
        }
        self.intermediate_data.clear();
        Ok(old_buffer)
    }
}
