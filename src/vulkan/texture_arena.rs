use std::{mem::ManuallyDrop, sync::Arc};

use anyhow::{Context, Result};
use ash::{
    prelude::VkResult,
    vk::{self, Handle},
};
use gpu_allocator::{MemoryLocation, vulkan::Allocation};
use slotmap::{SecondaryMap, SlotMap};

use crate::utils::align_to;

use super::Device;

const SAMPLER_SET: u32 = 0;
const IMAGE_SET: u32 = 1;
const STORAGE_SET: u32 = 0;

const LINEAR_EDGE_SAMPLER_IDX: u32 = 0;
const LINEAR_BORDER_SAMPLER_IDX: u32 = 1;
const NEAREST_SAMPLER_IDX: u32 = 2;

pub const DUMMY_IMAGE_IDX: usize = 0;
pub const DITHER_IMAGE_IDX: usize = 1;
pub const NOISE_IMAGE_IDX: usize = 2;
pub const BLUE_IMAGE_IDX: usize = 3;

const IMAGES_COUNT: u32 = 2048;
const STORAGE_COUNT: u32 = 2048;
const SAMPLER_COUNT: u32 = 8;

#[derive(Debug, Clone, Copy)]
pub struct ImageDimensions {
    pub width: usize,
    pub height: usize,
    pub padded_bytes_per_row: usize,
    pub unpadded_bytes_per_row: usize,
}

impl ImageDimensions {
    pub fn new(width: usize, height: usize, alignment: u64) -> Self {
        let channel_width = std::mem::size_of::<[u8; 4]>();
        let unpadded_bytes_per_row = width * channel_width;
        let padded_bytes_per_row = align_to(unpadded_bytes_per_row, alignment as usize);
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

pub struct ManagedImage {
    pub image: vk::Image,
    pub memory: ManuallyDrop<Allocation>,
    pub image_dimensions: ImageDimensions,
    pub data: Option<&'static mut [u8]>,
    pub format: vk::Format,
    device: Arc<Device>,
}

impl ManagedImage {
    pub fn new(
        device: &Arc<Device>,
        info: &vk::ImageCreateInfo,
        usage: MemoryLocation,
    ) -> anyhow::Result<Self> {
        let (image, memory) = device.create_image(info, usage)?;
        let memory_reqs = unsafe { device.get_image_memory_requirements(image) };
        let image_dimensions = ImageDimensions::new(
            info.extent.width as _,
            info.extent.height as _,
            memory_reqs.alignment,
        );
        Ok(Self {
            image,
            memory: ManuallyDrop::new(memory),
            image_dimensions,
            format: info.format,
            data: None,
            device: device.clone(),
        })
    }

    pub fn map_memory(&mut self) -> Option<&mut [u8]> {
        self.memory.mapped_slice_mut()
    }
}

impl Drop for ManagedImage {
    fn drop(&mut self) {
        unsafe {
            let memory = ManuallyDrop::take(&mut self.memory);
            self.device.destroy_image(self.image, memory);
        }
    }
}

const MAX_MIPCOUNT: usize = 8;

slotmap::new_key_type! {
    pub struct ImageHandle;
}

pub enum ScreenRelation {
    Identity,
    Half,
    Quarter,
    None,
}

impl ScreenRelation {
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Identity => Some(1.),
            Self::Half => Some(0.5),
            Self::Quarter => Some(0.25),
            Self::None => None,
        }
    }
}

// TODO: Name Images
#[derive(Debug)]
pub struct Image {
    name: Option<String>,
    pub inner: vk::Image,
    pub views: [Option<vk::ImageView>; MAX_MIPCOUNT],
    pub info: Option<vk::ImageCreateInfo<'static>>,
    memory: Option<Allocation>,
}

impl Image {
    pub fn name(&self) -> Option<&str> {
        match self.name.as_ref() {
            Some(name) => Some(name),
            None => None,
        }
    }

    fn destroy(&mut self, device: &Device) {
        if let Some(memory) = self.memory.take() {
            device.destroy_image(self.inner, memory);
            for view in self
                .views
                .iter()
                .filter_map(|view| view.as_ref())
                .filter(|view| !view.is_null())
            {
                unsafe { device.destroy_image_view(*view, None) };
            }
        }
    }
}

pub struct TextureArena {
    pub images: SlotMap<ImageHandle, Image>,
    pub sampled_indices: SecondaryMap<ImageHandle, [Option<u32>; MAX_MIPCOUNT]>,
    pub storage_indices: SecondaryMap<ImageHandle, [Option<u32>; MAX_MIPCOUNT]>,
    last_sampled_idx: u32,
    last_storage_idx: u32,

    screen_sized_images: SecondaryMap<ImageHandle, f32>,
    default_images: Vec<ImageHandle>,

    pub samplers: [vk::Sampler; SAMPLER_COUNT as usize],

    pub sampled_set: vk::DescriptorSet,
    pub sampled_set_layout: vk::DescriptorSetLayout,
    pub storage_set: vk::DescriptorSet,
    pub storage_set_layout: vk::DescriptorSetLayout,

    descriptor_pool: vk::DescriptorPool,
    device: Arc<Device>,
}

impl TextureArena {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(IMAGES_COUNT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(STORAGE_COUNT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(SAMPLER_COUNT),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(
                        vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                            | vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                    )
                    .pool_sizes(&pool_sizes)
                    .max_sets(2),
                None,
            )?
        };
        device.name_object(descriptor_pool, "Texture Pool");

        // Sampled textures
        let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND;
        let binding_flags = [
            binding_flags,
            binding_flags
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
        ];
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let sampler_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(SAMPLER_SET)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(SAMPLER_COUNT);
        let image_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(IMAGE_SET)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(
                device
                    .descriptor_indexing_props
                    .max_descriptor_set_update_after_bind_sampled_images,
            );
        let bindings = [sampler_set_layout_binding, image_set_layout_binding];
        let sampled_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )?
        };

        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&[IMAGES_COUNT]);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&sampled_set_layout))
            .push_next(&mut variable_info);
        let sampled_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? }[0];
        device.name_object(sampled_set, "Sampled Images Set");

        // Storage textures
        let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(std::slice::from_ref(&binding_flags));
        let storage_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(STORAGE_SET)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(
                device
                    .descriptor_indexing_props
                    .max_descriptor_set_update_after_bind_storage_images,
            );
        let bindings = [storage_set_layout_binding];
        let storage_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )?
        };
        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&[STORAGE_COUNT]);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&storage_set_layout))
            .push_next(&mut variable_info);
        let storage_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? }[0];
        device.name_object(storage_set, "Storage Images Set");

        let mut samplers = [vk::Sampler::null(); SAMPLER_COUNT as usize];
        let mut sampler_create_info = vk::SamplerCreateInfo::default()
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None)? };
        let descriptor_image_info = vk::DescriptorImageInfo::default().sampler(sampler);
        let mut desc_write = vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .dst_set(sampled_set)
            .dst_binding(SAMPLER_SET)
            .image_info(std::slice::from_ref(&descriptor_image_info))
            .dst_array_element(LINEAR_EDGE_SAMPLER_IDX);
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[LINEAR_EDGE_SAMPLER_IDX as usize] = sampler;

        sampler_create_info = sampler_create_info
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None)? };
        let descriptor_image_info = vk::DescriptorImageInfo::default().sampler(sampler);
        desc_write = desc_write
            .dst_array_element(LINEAR_BORDER_SAMPLER_IDX)
            .image_info(std::slice::from_ref(&descriptor_image_info));
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[LINEAR_BORDER_SAMPLER_IDX as usize] = sampler;

        sampler_create_info = sampler_create_info
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None)? };
        let descriptor_image_info = vk::DescriptorImageInfo::default().sampler(sampler);
        desc_write = desc_write
            .dst_array_element(NEAREST_SAMPLER_IDX)
            .image_info(std::slice::from_ref(&descriptor_image_info));
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[NEAREST_SAMPLER_IDX as usize] = sampler;

        let mut texture_arena = Self {
            images: SlotMap::with_key(),
            sampled_indices: SecondaryMap::new(),
            storage_indices: SecondaryMap::new(),

            last_sampled_idx: 0,
            last_storage_idx: 0,

            screen_sized_images: SecondaryMap::new(),
            default_images: vec![],

            samplers,

            sampled_set,
            sampled_set_layout,
            storage_set,
            storage_set_layout,

            descriptor_pool,

            device: device.clone(),
        };

        let image_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL);
        let handle = texture_arena.push_image(
            image_info,
            ScreenRelation::None,
            &[255, 255, 0, 255],
            Some("Dummy Image"),
        )?;
        texture_arena.default_images.push(handle);
        assert_eq!(
            DUMMY_IMAGE_IDX as u32,
            texture_arena.get_sampled_idx(handle, 0)
        );

        let mut push_dds = |bytes: &[u8], name| -> Result<()> {
            let dds = ddsfile::Dds::read(bytes)?;
            let info = vk::ImageCreateInfo::default()
                .extent(vk::Extent3D {
                    width: dds.get_width(),
                    height: dds.get_height(),
                    depth: 1,
                })
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .tiling(vk::ImageTiling::OPTIMAL);
            let handle = texture_arena.push_image(
                info,
                ScreenRelation::None,
                dds.get_data(0)?,
                Some(name),
            )?;
            texture_arena.default_images.push(handle);
            texture_arena
                .device
                .name_object(texture_arena.images[handle].inner, name);
            Ok(())
        };

        push_dds(include_bytes!("../../assets/dither.dds"), "Dither Image")?;
        push_dds(include_bytes!("../../assets/noise.dds"), "Noise Image")?;
        push_dds(
            include_bytes!("../../assets/BLUE_RGBA_0.dds"),
            "Blue Noise Image",
        )?;

        Ok(texture_arena)
    }

    pub fn get_image(&self, handle: ImageHandle) -> &Image {
        &self.images[handle]
    }

    pub fn get_image_mut(&mut self, handle: ImageHandle) -> &mut Image {
        &mut self.images[handle]
    }

    pub fn get_sampled_idx(&mut self, handle: ImageHandle, mip_level: u32) -> u32 {
        if let Some(info) = self.images[handle].info {
            assert!(mip_level < info.mip_levels);
        }
        match self.sampled_indices[handle][mip_level as usize] {
            Some(idx) => idx,
            None => {
                let image = &mut self.images[handle];
                let view = match image.views[mip_level as usize] {
                    Some(view) => view,
                    None => {
                        let view = make_image_view(
                            &self.device,
                            &image.inner,
                            image.info.unwrap().format,
                            mip_level,
                        )
                        .unwrap();
                        image.views[mip_level as usize] = Some(view);
                        view
                    }
                };

                let sampled_idx = self.last_sampled_idx;
                update_sampled_set(&self.device, &self.sampled_set, sampled_idx, &view);
                self.sampled_indices[handle][mip_level as usize] = Some(sampled_idx);
                self.last_sampled_idx += 1;

                sampled_idx
            }
        }
    }

    pub fn get_storage_idx(&mut self, handle: ImageHandle, mip_level: u32) -> u32 {
        if let Some(info) = self.images[handle].info {
            assert!(info.usage.contains(vk::ImageUsageFlags::STORAGE));
            assert!(mip_level < info.mip_levels);
        }
        match self.storage_indices[handle][mip_level as usize] {
            Some(idx) => idx,
            None => {
                let image = &mut self.images[handle];
                let view = match image.views[mip_level as usize] {
                    Some(view) => view,
                    None => {
                        let view = make_image_view(
                            &self.device,
                            &image.inner,
                            image.info.unwrap().format,
                            mip_level,
                        )
                        .unwrap();
                        image.views[mip_level as usize] = Some(view);
                        view
                    }
                };

                let storage_idx = self.last_storage_idx;
                update_storage_set(&self.device, &self.storage_set, storage_idx, &view);
                self.storage_indices[handle][mip_level as usize] = Some(storage_idx);
                self.last_storage_idx += 1;

                storage_idx
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        for (handle, &factor) in self.screen_sized_images.iter() {
            let image = &mut self.images[handle];
            let Some(info) = image.info.as_mut() else {
                continue;
            };
            let usage = info.usage;
            info.extent.width = (factor * width as f32) as u32;
            info.extent.height = (factor * height as f32) as u32;

            let (new_image, new_memory) =
                self.device.create_image(info, MemoryLocation::GpuOnly)?;
            image.destroy(&self.device);
            image.inner = new_image;
            image.memory = Some(new_memory);
            if let Some(name) = image.name.as_ref() {
                self.device.name_object(image.inner, name);
            }

            for (mip_level, view) in image
                .views
                .iter_mut()
                .enumerate()
                .filter_map(|(i, view)| view.as_mut().map(|v| (i, v)))
            {
                *view = make_image_view(
                    &self.device,
                    &image.inner,
                    image.info.unwrap().format,
                    mip_level as u32,
                )?;
                if let Some(name) = image.name.as_ref() {
                    self.device
                        .name_object(*view, &format!("{name} View {mip_level}"));
                }

                if let Some(idx) = self.sampled_indices[handle][mip_level] {
                    update_sampled_set(&self.device, &self.sampled_set, idx, view);
                }
                if usage.contains(vk::ImageUsageFlags::STORAGE) {
                    if let Some(idx) = self.storage_indices[handle][mip_level] {
                        update_storage_set(&self.device, &self.storage_set, idx, view);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn push_image(
        &mut self,
        mut info: vk::ImageCreateInfo<'static>,
        screen_relation: ScreenRelation,
        data: &[u8],
        name: Option<&str>,
    ) -> Result<ImageHandle> {
        if !data.is_empty() {
            info.usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if let Some(factor) = screen_relation.as_f32() {
            info.extent.width = (factor * info.extent.width as f32) as u32;
            info.extent.height = (factor * info.extent.height as f32) as u32;
        }
        let (image, memory) = self.device.create_image(&info, MemoryLocation::GpuOnly)?;
        if let Some(name) = name {
            self.device.name_object(image, name);
        }

        if !data.is_empty() {
            let mut staging = self.device.create_buffer(
                memory.size(),
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )?;
            let mapped = staging.map_memory().context("Failed to map memory")?;
            mapped[..data.len()].copy_from_slice(data);

            self.device.one_time_submit(|device, cbuff| unsafe {
                device.image_transition(
                    &cbuff,
                    &image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
                let regions = vk::BufferImageCopy::default()
                    .image_extent(info.extent)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                        mip_level: 0,
                    });
                device.cmd_copy_buffer_to_image(
                    cbuff,
                    staging.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[regions],
                );
                device.image_transition(
                    &cbuff,
                    &image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
                Ok(())
            })?;
        }

        let mut views = [None; MAX_MIPCOUNT];
        let mut sampled_indices = [None; MAX_MIPCOUNT];
        let mut storage_indices = [None; MAX_MIPCOUNT];
        for (i, view) in views.iter_mut().enumerate().take(info.mip_levels as usize) {
            let new_view = make_image_view(&self.device, &image, info.format, i as u32)?;
            *view = Some(new_view);
            if let Some(name) = name {
                self.device
                    .name_object(new_view, &format!("{name} View {i}"));
            }

            {
                let sampled_idx = self.last_sampled_idx;
                update_sampled_set(&self.device, &self.sampled_set, sampled_idx, &new_view);
                sampled_indices[i] = Some(sampled_idx);
                self.last_sampled_idx += 1;
            }
            if info.usage.contains(vk::ImageUsageFlags::STORAGE) {
                let storage_idx = self.last_storage_idx;
                update_storage_set(&self.device, &self.storage_set, storage_idx, &new_view);
                storage_indices[i] = Some(storage_idx);
                self.last_storage_idx += 1;
            }
        }

        let handle = self.images.insert(Image {
            inner: image,
            views,
            info: Some(info),
            memory: Some(memory),
            name: name.map(|name| name.to_owned()),
        });
        self.sampled_indices.insert(handle, sampled_indices);
        self.storage_indices.insert(handle, storage_indices);

        if let Some(factor) = screen_relation.as_f32() {
            self.screen_sized_images.insert(handle, factor);
        }

        Ok(handle)
    }

    pub fn push_external_image(
        &mut self,
        image: vk::Image,
        view: vk::ImageView,
    ) -> Result<ImageHandle> {
        let mut views = [None; MAX_MIPCOUNT];
        views[0] = Some(view);
        let handle = self.images.insert(Image {
            inner: image,
            views,
            info: None,
            memory: None,
            name: None,
        });

        {
            let sampled_idx = self.last_sampled_idx;
            update_sampled_set(&self.device, &self.sampled_set, sampled_idx, &view);
            let mut indices = [None; MAX_MIPCOUNT];
            indices[0] = Some(sampled_idx);
            self.sampled_indices.insert(handle, indices);
            self.last_sampled_idx += 1;
        }

        {
            let storage_idx = self.last_storage_idx;
            update_storage_set(&self.device, &self.storage_set, storage_idx, &view);
            let mut indices = [None; MAX_MIPCOUNT];
            indices[0] = Some(storage_idx);
            self.storage_indices.insert(handle, indices);
            self.last_storage_idx += 1;
        }

        Ok(handle)
    }

    pub fn update_external_image(
        &mut self,
        handle: ImageHandle,
        new_image: vk::Image,
        new_view: vk::ImageView,
    ) {
        let image = &mut self.images[handle];
        image.inner = new_image;
        image.views[0] = Some(new_view);

        let sampled_idx = self.get_sampled_idx(handle, 0);
        update_sampled_set(&self.device, &self.sampled_set, sampled_idx, &new_view);
        let storage_idx = self.get_storage_idx(handle, 0);
        update_storage_set(&self.device, &self.storage_set, storage_idx, &new_view);
    }
}

fn update_sampled_set(device: &Device, set: &vk::DescriptorSet, idx: u32, view: &vk::ImageView) {
    let image_info = vk::DescriptorImageInfo::default()
        .image_view(*view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(*set)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .dst_binding(IMAGE_SET)
        .image_info(std::slice::from_ref(&image_info))
        .dst_array_element(idx);
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

fn update_storage_set(device: &Device, set: &vk::DescriptorSet, idx: u32, view: &vk::ImageView) {
    let image_info = vk::DescriptorImageInfo::default()
        .image_view(*view)
        .image_layout(vk::ImageLayout::GENERAL);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(*set)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .dst_binding(STORAGE_SET)
        .image_info(std::slice::from_ref(&image_info))
        .dst_array_element(idx);
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

fn make_image_view(
    device: &Device,
    image: &vk::Image,
    format: vk::Format,
    base_mip_level: u32,
) -> VkResult<vk::ImageView> {
    unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .image(*image)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(base_mip_level)
                        .level_count(if base_mip_level == 0 {
                            vk::REMAINING_MIP_LEVELS
                        } else {
                            1
                        })
                        .base_array_layer(0)
                        .layer_count(1),
                ),
            None,
        )
    }
}

impl Drop for TextureArena {
    fn drop(&mut self) {
        unsafe {
            self.images
                .values_mut()
                .for_each(|image| image.destroy(&self.device));

            self.samplers
                .iter()
                .for_each(|&sampler| self.device.destroy_sampler(sampler, None));
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[self.sampled_set, self.storage_set]);
            self.device
                .destroy_descriptor_set_layout(self.sampled_set_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.storage_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
