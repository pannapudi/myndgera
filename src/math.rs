use glam::{Mat4, Vec3, Vec3Swizzles, Vec4, vec3};

#[allow(non_upper_case_globals)]
pub const sin: fn(f32) -> f32 = f32::sin;
#[allow(non_upper_case_globals)]
pub const cos: fn(f32) -> f32 = f32::cos;

pub trait VecMap {
    fn map(self, f: impl FnMut(f32) -> f32) -> Self;
}

impl VecMap for Vec3 {
    fn map(self, mut f: impl FnMut(f32) -> f32) -> Self {
        Self::new(f(self.x), f(self.y), f(self.z))
    }
}
impl VecMap for Vec4 {
    fn map(self, mut f: impl FnMut(f32) -> f32) -> Self {
        Self::new(f(self.x), f(self.y), f(self.z), f(self.w))
    }
}

pub fn erot(p: Vec3, ax: Vec3, a: f32) -> Vec3 {
    Vec3::lerp(ax.dot(p) * ax, p, a.cos()) + ax.cross(p) * a.sin()
}

pub fn hash11(mut p: f32) -> f32 {
    p = (p * 0.1031).fract();
    p *= p + 33.33;
    p *= p + p;
    p.fract()
}

pub fn hash13(p: f32) -> Vec3 {
    let mut p3 = (vec3(p, p, p) * vec3(0.1031, 0.1030, 0.0973)).fract();
    p3 += p3.dot(p3.yzx() + 33.33);
    ((p3.xxy() + p3.yzz()) * p3.zyx()).fract()
}

pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1. - t) + b * t
}

pub fn smoothstep(x: f32, edge0: f32, edge1: f32) -> f32 {
    let x = ((x - edge0) / (edge1 - edge0)).clamp(0., 1.);
    x * x * (3. - 2. * x)
}

// https://www.shadertoy.com/view/md2GWW
pub fn step_noise(x: f32, n: f32) -> f32 {
    let i = x.floor();
    let s = 0.1;
    let u = smoothstep(0.5 - s, 0.5 + s, x.fract());
    mix((hash11(i) * n).floor(), (hash11(i + 1.) * n).floor(), u) // from 0. to n - 1.
}

pub fn smooth_floor(x: f32, c: f32) -> f32 {
    let a = x.fract();
    let b = x.floor();
    b + (a.powf(c) - (1. - a).powf(c)) / 2.
}

pub fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
    let dir = target - eye;
    let z_axis = dir.normalize();
    let x_axis = z_axis.cross(vec3(0., 1., 0.)).normalize();
    let y_axis = x_axis.cross(z_axis);

    Mat4::from_cols(
        x_axis.extend(0.),
        y_axis.extend(0.),
        z_axis.extend(0.),
        eye.extend(1.),
    )
}
