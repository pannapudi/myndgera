use anyhow::Result;
use ash::{
    prelude::VkResult,
    vk::{self, Extent2D},
};
use dolly::prelude::YawPitch;
use glam::{Vec2, Vec3, vec2, vec3};
use myndgera::{
    App, AppState, Camera, FIXED_TIME_STEP, Framework, KeyboardMap, RenderContext,
    vulkan::{
        FragmentOutputDesc, FragmentShaderDesc, FrameGuard, RenderHandle, VertexInputDesc,
        VertexShaderDesc,
    },
};
use std::error::Error;
use winit::{event_loop::EventLoop, keyboard::KeyCode};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PushConstant {
    resolution: Vec2,
    pos: Vec3,
    mouse: Vec2,
    mouse_pressed: u32,
    time: f32,
    time_delta: f32,
    frame: u32,
    camera_buffer: u64,
}

struct Trig {
    push_constant: PushConstant,
    render_pipeline: RenderHandle,
}

impl Framework for Trig {
    fn init(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        state.camera = Camera::new(vec3(0., 0., -3.), 180.0, 0.0);

        let push_constant = PushConstant {
            pos: Vec3::from([0.; 3]),
            resolution: vec2(
                ctx.swapchain.extent.width as f32,
                ctx.swapchain.extent.height as f32,
            ),
            mouse: state.input.mouse_state.screen_position,
            mouse_pressed: state.input.mouse_state.left_pressed() as u32,
            time: state.time,
            frame: state.frame,
            time_delta: 1. / 60.,
            camera_buffer: state.camera_uniform_gpu.address,
        };
        let vertex_shader_desc = VertexShaderDesc {
            shader_path: "examples/toy/shader.vert".into(),
            ..Default::default()
        };
        let fragment_shader_desc = FragmentShaderDesc {
            shader_path: "examples/toy/shader.frag".into(),
            ..Default::default()
        };
        let fragment_output_desc = FragmentOutputDesc {
            surface_format: ctx.swapchain.format,
            ..Default::default()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<PushConstant>() as _)
            .stage_flags(
                vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::COMPUTE,
            );
        let render_pipeline = state.pipeline_arena.create_render_pipeline(
            VertexInputDesc::default(),
            vertex_shader_desc,
            fragment_shader_desc,
            fragment_output_desc,
            &[push_constant_range],
            &[state.texture_arena.sampled_set_layout],
        )?;

        state.key_map = {
            use winit::keyboard::KeyCode::*;
            KeyboardMap::new()
                .bind(KeyW, ("move_fwd", 1.0))
                .bind(KeyS, ("move_fwd", -1.0))
                .bind(KeyD, ("move_right", 1.0))
                .bind(KeyA, ("move_right", -1.0))
                .bind(KeyQ, ("move_up", -1.0))
                .bind(KeyE, ("move_up", 1.0))
                .bind(ShiftLeft, ("boost", 1.0))
                .bind(ControlLeft, ("boost", -1.0))
        };

        Ok(Self {
            push_constant,
            render_pipeline,
        })
    }

    fn draw(
        &mut self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &mut FrameGuard,
    ) -> VkResult<()> {
        frame.begin_rendering(
            &ctx.swapchain.get_image(frame.image_idx),
            &ctx.swapchain.get_view(frame.image_idx),
            vk::AttachmentLoadOp::CLEAR,
            [1., 1., 1., 1.],
        );

        let pipeline = state.pipeline_arena.get_pipeline(self.render_pipeline);
        frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pipeline.pipeline);
        frame.bind_push_constants(
            pipeline.layout,
            vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::COMPUTE,
            &[self.push_constant],
        );
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout,
            &[state.texture_arena.sampled_set],
        );
        frame.draw(3, 1, 0, 0);

        frame.end_rendering();

        Ok(())
    }

    fn update(
        &mut self,
        ctx: &RenderContext,
        state: &mut AppState,
        _cbuff: &vk::CommandBuffer,
    ) -> Result<()> {
        // dbg!(&state.camera);
        state.input.process_position(&mut self.push_constant.pos);
        let Extent2D { width, height } = ctx.swapchain.extent;
        self.push_constant.resolution.x = width as f32;
        self.push_constant.resolution.y = height as f32;
        self.push_constant.time = state.time;
        self.push_constant.frame = state.frame;
        self.push_constant.time_delta = 1. / 60.;
        self.push_constant.mouse = state.input.mouse_state.screen_position / 2.;
        self.push_constant.mouse_pressed = state.input.mouse_state.left_held() as u32;

        if state.input.mouse_state.left_held() {
            let sensitivity = 0.5;
            state.camera.rig.driver_mut::<YawPitch>().rotate_yaw_pitch(
                -sensitivity * state.input.mouse_state.delta.x,
                -sensitivity * state.input.mouse_state.delta.y,
            );
        }

        let dt = FIXED_TIME_STEP as f32;
        let key_map = state.key_map.map(&state.input.keyboard_state);
        let translation = Vec3::new(
            key_map["move_right"],
            key_map["move_up"],
            -key_map["move_fwd"],
        );

        let rotation: glam::Quat = state.camera.rig.final_transform.rotation.into();
        let move_vec = rotation * translation.clamp_length_max(1.0) * 4.0f32.powf(key_map["boost"]);

        state
            .camera
            .rig
            .driver_mut::<dolly::drivers::Position>()
            .translate(move_vec * dt * 5.0);

        let pos = self.push_constant.pos;
        if state.input.keyboard_state.was_just_pressed(KeyCode::F6) {
            println!("Posiiton: [{}, {}, {}]", pos.x, pos.y, pos.z);
            println!(
                "Mouse: [{}, {}]",
                self.push_constant.mouse.x, self.push_constant.mouse.y
            );
            let pos = state.camera.rig.final_transform.position;
            println!("Camera pos: {pos:?}");
            println!();
        }

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = &mut EventLoop::with_user_event();
    let event_loop = event_loop.build()?;

    let mut app = App::<Trig>::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;

    Ok(())
}
