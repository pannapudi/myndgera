use anyhow::{Result, bail};
use ash::{prelude::VkResult, vk};
use either::Either;
use glam::{vec2, vec3};
use gpu_allocator::MemoryLocation;
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{error, level_filters::LevelFilter, warn};
use tracing_subscriber::{filter::EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize, PhysicalSize},
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes, WindowId},
};

mod camera;
mod input;
pub mod math;
pub mod passes;
mod recorder;
mod render_context;
mod shader_compiler;
pub mod utils;
pub mod vulkan;
mod watcher;

pub use shader_compiler::*;
pub use watcher::Watcher;

use self::recorder::Recorder;
pub use self::{
    camera::{Camera, CameraUniform},
    input::{Input, KeyboardMap},
    render_context::RenderContext,
    utils::*,
    vulkan::*,
};

pub const UPDATES_PER_SECOND: u32 = 60;
pub const FIXED_TIME_STEP: f64 = 1. / UPDATES_PER_SECOND as f64;
pub const MAX_FRAME_TIME: f64 = 15. * FIXED_TIME_STEP; // 0.25;

pub const COLOR_SUBRESOURCE_MASK: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: vk::REMAINING_MIP_LEVELS,
    base_array_layer: 0,
    layer_count: vk::REMAINING_ARRAY_LAYERS,
};

pub const SHADER_FOLDER: &str = "shaders";
pub const VIDEO_FOLDER: &str = "recordings";
pub const SCREENSHOT_FOLDER: &str = "screenshots";

#[derive(Debug)]
pub enum UserEvent {
    Glsl { path: std::path::PathBuf },
}

pub trait Framework: Sized {
    fn name() -> &'static str {
        "Myndgera"
    }
    fn init(app: &RenderContext, _ctx: &mut AppState) -> Result<Self>;
    fn resize(&mut self, _ctx: &mut RenderContext) -> Result<()> {
        Ok(())
    }
    fn update(
        &mut self,
        _ctx: &RenderContext,
        _state: &mut AppState,
        _cbuff: &vk::CommandBuffer,
    ) -> Result<()> {
        Ok(())
    }
    fn draw(
        &mut self,
        _ctx: &RenderContext,
        _state: &mut AppState,
        _frame: &mut FrameGuard,
    ) -> VkResult<()> {
        Ok(())
    }
}

pub struct AppState {
    pub frame: u32,

    pub staging_write: StagingWrite,

    pub pipeline_arena: PipelineArena,
    pub texture_arena: TextureArena,
    pub swapchain_handles: Vec<ImageHandle>,

    pub input: Input,
    pub key_map: KeyboardMap,

    pub camera: Camera,
    pub camera_uniform: CameraUniform,
    pub camera_uniform_gpu: BufferTyped<CameraUniform>,

    recorder: Recorder,
    recording_time: Option<Duration>,

    pub pause: bool,
    pub time: f32,
    pub timeline: Instant,
    pub backup_time: Duration,
    frame_instant: Instant,
    frame_accumulated_time: f64,
}

impl AppState {
    fn new(
        ctx: &mut RenderContext,
        proxy: EventLoopProxy<UserEvent>,
        recording_time: Option<Duration>,
    ) -> Result<Self> {
        let mut camera = Camera::new(vec3(0., 0., 10.), 0., 0.);
        camera.aspect = ctx.swapchain.extent.width as f32 / ctx.swapchain.extent.height as f32;
        let camera_uniform = camera.get_uniform(None);
        let camera_uniform_gpu = ctx.device.create_buffer_typed(
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::GpuOnly,
        )?;

        let recorder = Recorder::new();

        let staging_write = StagingWrite::new(&ctx.device)?;

        let file_watcher = Watcher::new(proxy)?;
        let pipeline_arena = PipelineArena::new(&ctx.device, file_watcher)?;

        let mut texture_arena = TextureArena::new(&ctx.device)?;
        let mut swapchain_handles = vec![];
        for (&image, &view) in ctx.swapchain.images.iter().zip(&ctx.swapchain.views) {
            swapchain_handles.push(texture_arena.push_external_image(image, view)?);
        }

        Ok(Self {
            frame: 0,

            camera,
            camera_uniform,
            camera_uniform_gpu,

            recorder,
            recording_time,

            pipeline_arena,
            texture_arena,
            swapchain_handles,

            staging_write,

            input: Input::new(),
            key_map: KeyboardMap::new(),

            pause: false,
            time: 0.,
            timeline: Instant::now(),
            backup_time: Duration::from_secs(0),
            frame_instant: Instant::now(),
            frame_accumulated_time: 0.,
        })
    }
}

pub struct AppInit<F> {
    framework: F,
    state: AppState,
    device: Arc<Device>,
    ctx: RenderContext,
}

impl<F> AppInit<F> {
    pub fn reload_shaders(&mut self, path: PathBuf) -> Result<()> {
        for frame in self.ctx.swapchain.frames.iter().filter_map(Option::as_ref) {
            let fences = std::slice::from_ref(&frame.present_finished);
            self.device.wait_for_fences(fences, true, u64::MAX)?;
        }

        let state = &mut self.state;
        let resolved = {
            let mapping = state.pipeline_arena.file_watcher.include_mapping.lock();
            mapping[Path::new(path.file_name().unwrap())].clone()
        };

        for ShaderSource { path, kind } in resolved {
            let handles = &state.pipeline_arena.path_mapping[&path];
            for handle in handles {
                let compiler = &state.pipeline_arena.shader_compiler;
                match handle {
                    Either::Left(handle) => {
                        let pipeline = &mut state.pipeline_arena.render_arena[*handle];
                        match kind {
                            ShaderKind::Vertex => pipeline.reload_vertex_lib(compiler, &path),
                            ShaderKind::Fragment => pipeline.reload_fragment_lib(compiler, &path),
                            ShaderKind::Compute => {
                                bail!("Supplied compute shader into the render pipeline!")
                            }
                        }?;
                        pipeline.link()?;
                    }
                    Either::Right(handle) => {
                        let pipeline = &mut state.pipeline_arena.compute_arena[*handle];
                        pipeline.reload(compiler)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<F: Framework> AppInit<F> {
    pub fn new(
        window: Window,
        proxy: EventLoopProxy<UserEvent>,
        recording_time: Option<Duration>,
    ) -> Result<Self> {
        let mut ctx = RenderContext::new(window)?;
        let mut state = AppState::new(&mut ctx, proxy, recording_time)?;
        let framework = F::init(&ctx, &mut state)?;
        let device = ctx.device.clone();

        Ok(Self {
            framework,
            state,
            ctx,
            device,
        })
    }

    fn update(&mut self) -> Result<()> {
        let state = &mut self.state;
        let new_instant = Instant::now();
        let frame_time = new_instant
            .duration_since(state.frame_instant)
            .as_secs_f64()
            .min(MAX_FRAME_TIME);
        state.frame_instant = new_instant;

        state.frame_accumulated_time += frame_time;
        while state.frame_accumulated_time >= FIXED_TIME_STEP {
            self.device.one_time_submit(|device, cbuff| {
                let _marker = device.create_scoped_marker(&cbuff, "State Update");

                state.input.tick();

                state.camera.rig.update(FIXED_TIME_STEP as f32);

                state.camera_uniform = state.camera.get_uniform(Some(&state.camera_uniform));
                state.staging_write.write_buffer(
                    state.camera_uniform_gpu.buffer,
                    bytemuck::bytes_of(&state.camera_uniform),
                );

                self.framework.update(&self.ctx, state, &cbuff)?;

                state.staging_write.consume_pending_writes(&cbuff)?;

                Ok(())
            })?;

            state.input.mouse_state.refresh();

            state.frame_accumulated_time -= FIXED_TIME_STEP;
        }
        Ok(())
    }

    pub fn recreate_swapchain(&mut self) -> Result<()> {
        if let Some(frame) = self.ctx.swapchain.get_current_frame() {
            let fences = std::slice::from_ref(&frame.present_finished);
            let one_second = Duration::from_secs(1).as_nanos() as u64;
            self.device.wait_for_fences(fences, true, one_second)?;
        }

        let PhysicalSize { width, height } = self.ctx.window.inner_size();
        self.ctx
            .swapchain
            .recreate(&self.ctx.surface, width, height)?;

        self.state.camera.aspect = width as f32 / height as f32;

        self.state.texture_arena.resize(width, height)?;

        let handles = &self.state.swapchain_handles;
        for ((&handle, &image), &view) in handles
            .iter()
            .zip(&self.ctx.swapchain.images)
            .zip(&self.ctx.swapchain.views)
        {
            self.state
                .texture_arena
                .update_external_image(handle, image, view);
        }

        self.framework.resize(&mut self.ctx)?;
        Ok(())
    }

    fn draw(&mut self) -> VkResult<()> {
        let mut frame = self.ctx.swapchain.acquire_next_image()?;

        self.framework
            .draw(&self.ctx, &mut self.state, &mut frame)?;

        self.ctx.window.pre_present_notify();

        if self.state.recorder.is_active() && self.state.recorder.ffmpeg_installed() {
            let res = self.device.capture_image_data(
                &self.ctx.swapchain.images[frame.image_idx],
                self.ctx.swapchain.extent(),
                |tex| self.state.recorder.record(tex),
            );
            if let Err(err) = res {
                error!("{err}");
            }
        }

        self.ctx.swapchain.submit_image(frame)?;

        self.state.frame = self.state.frame.wrapping_add(1);
        Ok(())
    }
}

impl<F: Framework> ApplicationHandler<UserEvent> for AppInit<F> {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.state.time = if !self.state.pause {
            self.state.timeline.elapsed().as_secs_f32()
        } else {
            self.state.backup_time.as_secs_f32()
        };

        let _ = self.update().map_err(|err| error!("{err}"));

        if let Some(limit) = self.state.recording_time {
            if self.state.timeline.elapsed() >= limit && self.state.recorder.is_active() {
                self.state.recorder.finish();
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.state.input.update_on_window_input(&event);
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(key),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let dt = Duration::from_secs_f32(1. / 60.);
                let state = &mut self.state;
                match key {
                    // NamedKey::F1 => print_help(),
                    NamedKey::F2 => {
                        if !state.pause {
                            state.backup_time = state.timeline.elapsed();
                        } else {
                            state.timeline = Instant::now() - state.backup_time;
                        }
                        state.pause = !state.pause;
                    }
                    NamedKey::F3 => {
                        if !state.pause {
                            state.backup_time = state.timeline.elapsed();
                            state.pause = true;
                        }
                        state.backup_time = state.backup_time.saturating_sub(dt);
                    }
                    NamedKey::F4 => {
                        if !state.pause {
                            state.backup_time = state.timeline.elapsed();
                            state.pause = true;
                        }
                        state.backup_time += dt;
                    }
                    NamedKey::F5 => {
                        // state.stats.pos = [0.; 3];
                        state.time = 0.;
                        state.frame = 0;
                        state.timeline = Instant::now();
                        state.backup_time = state.timeline.elapsed();
                    }
                    NamedKey::F7 => {
                        let _ = self
                            .device
                            .capture_image_data(
                                self.ctx.swapchain.get_current_image(),
                                self.ctx.swapchain.extent(),
                                |tex| state.recorder.screenshot(tex),
                            )
                            .map_err(|err| error!("{err}"));
                    }
                    NamedKey::F8 => {
                        if !state.recorder.is_active() {
                            let mut image_dimensions = self.ctx.swapchain.image_dimensions();
                            image_dimensions.width = align_to(image_dimensions.width, 2);
                            image_dimensions.height = align_to(image_dimensions.height, 2);
                            state.recorder.start(image_dimensions);
                        } else {
                            state.recorder.finish();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let scale_factor = self.ctx.window.scale_factor();
                let LogicalPosition { x, y } = position.to_logical::<f32>(scale_factor);
                if !self.state.pause {
                    let LogicalSize { width, height } =
                        self.ctx.window.inner_size().to_logical::<f32>(scale_factor);
                    let x = (x / width - 0.5) * 2.;
                    let y = -(y / height - 0.5) * 2.;
                    self.state.input.mouse_state.screen_position = vec2(x, y);
                }
            }
            WindowEvent::Resized { .. } => {
                if self.state.recorder.is_active() {
                    self.state.recorder.finish();
                }
                self.ctx.is_swapchain_dirty = true;
            }
            WindowEvent::RedrawRequested => {
                if self.ctx.is_swapchain_dirty {
                    match self.recreate_swapchain() {
                        Ok(_) => self.ctx.is_swapchain_dirty = false,
                        Err(err) => warn!("{err}"),
                    }
                }

                match self.draw() {
                    Ok(()) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        let _ = self.recreate_swapchain().map_err(|err| warn!("{err}"));
                        self.ctx.window.request_redraw();
                    }
                    Err(e) => panic!("error: {e}\n"),
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        self.state.input.update_on_device_input(event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.ctx.window.request_redraw();
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.state.recorder.close_thread();
        if let Some(handle) = self.state.recorder.thread_handle.take() {
            let _ = handle.join();
        }
        self.device.wait_idle();
        println!("// End from the loop. Bye bye~⏎ ");
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Glsl { path } => {
                match self.reload_shaders(path) {
                    Err(err) => eprintln!("{err}"),
                    Ok(()) => {
                        const ESC: &str = "\x1B[";
                        const RESET: &str = "\x1B[0m";
                        eprint!("\r{ESC}42m{ESC}K{RESET}\r");
                        std::io::stdout().flush().unwrap();
                        std::thread::spawn(|| {
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            eprint!("\r{ESC}40m{ESC}K{RESET}\r");
                            std::io::stdout().flush().unwrap();
                        });
                    }
                };
            }
        }
    }

    fn resumed(&mut self, _event_loop: &ActiveEventLoop) {
        panic!("We don't expect second call of `resumed` function on desktop.")
    }
}

#[allow(clippy::large_enum_variant)]
pub enum App<F> {
    Uninitialized { proxy: EventLoopProxy<UserEvent> },
    Init(AppInit<F>),
}

impl<F> App<F> {
    pub fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        Self::Uninitialized { proxy }
    }
}

impl<F: Framework> ApplicationHandler<UserEvent> for App<F> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::DEBUG.into())
                    .from_env_lossy(),
            )
            .init();

        let Args {
            recording_time,
            inner_size,
        } = parse_args().unwrap();

        let mut window_attributes = WindowAttributes::default().with_title(F::name());
        if let Some(size) = inner_size {
            window_attributes = window_attributes
                .with_resizable(false)
                .with_inner_size(LogicalSize::<u32>::from(size));
        }

        println!("// Set up our new world⏎ ");
        println!("// And let's begin the⏎ ");
        println!("\tSIMULATION⏎ \n");

        event_loop.set_control_flow(ControlFlow::Poll);
        if let Self::Uninitialized { proxy } = self {
            let window = event_loop
                .create_window(window_attributes)
                .expect("Failed to create window");
            *self = Self::Init(
                AppInit::new(window, proxy.clone(), recording_time)
                    .expect("Failed to initialize application"),
            );
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Self::Init(app) = self {
            app.window_event(event_loop, window_id, event);
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        if let Self::Init(app) = self {
            app.new_events(event_loop, cause);
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        if let Self::Init(app) = self {
            app.user_event(event_loop, event);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Self::Init(app) = self {
            app.device_event(event_loop, device_id, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.about_to_wait(event_loop);
        }
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.suspended(event_loop);
        }
    }

    fn exiting(&mut self, event_loop: &ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.exiting(event_loop);
        }
    }

    fn memory_warning(&mut self, event_loop: &ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.memory_warning(event_loop);
        }
    }
}
