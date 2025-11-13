#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 final_color;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
#include "textures.glsl"
vec4 Tex(uint tex_id, uint smp_id) { return Tex(tex_id, smp_id, in_uv); }
vec4 TexLinear(uint tex_id) { return Tex(tex_id, LINEAR_SAMPL, in_uv); }
vec4 TexNear(uint tex_id) { return Tex(tex_id, NEAREST_SAMPL, in_uv); }

#include <camera.glsl>
#include <prelude.glsl>

layout(scalar, push_constant) uniform PushConstant {
    vec2 resolution;
    vec3 pos;
    vec2 mouse;
    bool mouse_pressed;
    float time;
    float dt;
    uint frame;
    CameraBuf camera;
}
pc;

const int MAT_WAVE = 2;

struct Hit {
    float t;
    float d;
    int mat;
};

Hit hit_default() { return Hit(0.001, 1e9, -1); }
Hit hit_init(float dist, int mat) { return Hit(0.001, dist, mat); }
Hit hit_init(float t, float dist, int mat) { return Hit(t, dist, mat); }
Hit min(Hit a, Hit b) {
    if (a.d < b.d) {
        return a;
    }
    return b;
}

const float MDIST = 100.;
const int ITERS_TRACE = 9;
const int ITERS_NORM = 20;

const float SCRL_SPEED = 1.5;
const vec2 SCRL_DIR = vec2(1.0, 1.2);
const float HOR_SCALE = 1.1;
const float FREQ_SCALE = 1.28;
const float TIME_SCALE = 1.095;
const float WEIGHT_SCALE = 0.8;
const float DRAG = 0.9;
const float HEIGHT_DIV = 2.3;

const float WAVE_ROT_ANGLE = 6.21;
const mat2 WAVE_ROT =
    mat2(cos(WAVE_ROT_ANGLE), -sin(WAVE_ROT_ANGLE), sin(WAVE_ROT_ANGLE), cos(WAVE_ROT_ANGLE));
const float WAVE_FREQ = 0.6;
const float OCC_SPEED = 1.4;
const float DX_DET = 0.65;

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

float sat(float x) { return clamp(x, 0., 1.); }
vec3 sat(vec3 x) { return clamp(x, 0., 1.); }

float sd_plane(vec3 p, vec3 off, vec3 n) { return dot(p - off, n); }

vec2 sd_wave_diff(vec2 wave_pos, int iter_num, float t) {
    vec2 res = vec2(0.0);
    vec2 wave_dir = vec2(1., 0.);
    float wave_weight = 1.0;
    wave_pos += t * SCRL_SPEED * SCRL_DIR;
    wave_pos *= HOR_SCALE;
    float wave_freq = WAVE_FREQ;
    float wave_time = OCC_SPEED * t;
    for (int i = 0; i < iter_num; ++i) {
        wave_dir *= WAVE_ROT;
        float x = dot(wave_dir, wave_pos) * wave_freq + wave_time;
        float dx = exp(sin(x) - 1.0) * cos(x) * wave_weight;
        res += dx * wave_dir / pow(wave_weight, DX_DET);

        wave_freq *= FREQ_SCALE;
        wave_time *= TIME_SCALE;
        wave_pos -= wave_dir * dx * DRAG;
        wave_weight *= WEIGHT_SCALE;
    }

    float wave_sum = -(pow(WEIGHT_SCALE, float(iter_num)) - 1.) * HEIGHT_DIV;
    return res / pow(wave_sum, 1. - DX_DET);
}

vec3 sd_wave_normal(vec3 p, float t) {
    vec2 wavedx = -sd_wave_diff(p.xz, ITERS_NORM, t);
    return normalize(vec3(wavedx.x, 1.0, wavedx.y));
}

float sd_wave(vec2 wave_pos, int iter_num, float t) {
    float res = 0.0;
    vec2 wave_dir = vec2(1., 0.);
    float wave_weight = 1.0;
    wave_pos += t * SCRL_SPEED * SCRL_DIR;
    wave_pos *= HOR_SCALE;
    float wave_freq = WAVE_FREQ;
    float wave_time = OCC_SPEED * t;
    for (int i = 0; i < iter_num; ++i) {
        wave_dir *= WAVE_ROT;
        float x = dot(wave_dir, wave_pos) * wave_freq + wave_time;
        float wave = exp(sin(x) - 1.0) * wave_weight;
        res += wave;

        wave_freq *= FREQ_SCALE;
        wave_time *= TIME_SCALE;
        wave_pos -= wave_dir * wave * DRAG * cos(x);
        wave_weight *= WEIGHT_SCALE;
    }

    float wave_sum = -(pow(WEIGHT_SCALE, float(iter_num)) - 1.) * HEIGHT_DIV;
    return res / wave_sum;
}

Hit map(vec3 p) {
    float t = pc.time;
    Hit hit = hit_default();

    float wave = sd_wave(p.xz, ITERS_TRACE, t);
    float plane = sd_plane(p, vec3(0.), vec3(0., 1., 0.));
    hit = min(hit, hit_init(plane - wave, 2));

    return hit;
}

vec3 get_norm(vec3 p, Hit hit, float t) {
    if (hit.mat == MAT_WAVE) {
        return sd_wave_normal(p, t);
    } else {
        mat3 k = mat3(p, p, p) - mat3(0.0001);
        return normalize(hit.d - vec3(map(k[0]).d, map(k[1]).d, map(k[2]).d));
    }
}

Hit trace(vec3 ro, vec3 rd) {
    float t = 0.0;
    for (int i = 0; i < 100; ++i) {
        vec3 pos = ro + rd * t;
        Hit hit = map(pos);
        float d = hit.d;
        if (d < 0.001) {
            return hit_init(t, d, hit.mat);
        }
        t += d;
        if (t > MDIST) {
            break;
        }
    }
    return hit_init(t, 1e9, -1);
}

mat3 get_cam(vec3 eye, vec3 target) {
    vec3 frw = normalize(target - eye);
    vec3 up = normalize(cross(frw, vec3(0., 1., 0.)));
    vec3 side = cross(frw, up);
    return mat3(up, frw, side);
}

vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) { return a + b * cos(2.0 * PI * (c * t + d)); }
vec3 spc(float n, float bright) {
    return pal(n, vec3(bright), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67));
}
vec2 sunrot = vec2(-0.3, 0.10);

vec3 sky(vec3 rd) {
    float rad = 0.075;
    vec3 col = vec3(0.);

    float sky_palette = 0.08;
    rd.yz *= rot(sunrot.y);
    rd.xz *= rot(sunrot.x);
    float px = min(fwidth(rd).x, fwidth(rd).y);
    float sFade = px * 2.;
    float zFade = rd.z * 0.5 + 0.5;

    vec3 sc = spc(sky_palette - 0.1, 0.6) * 0.85;
    float a = length(rd.xy);
    vec3 sun = smoothstep(a - px - sFade, a + px + sFade, rad) * sc * zFade * 3.;
    col += sun;
    col += rad / (rad + pow(a, 1.7)) * sc * zFade;
    col = col + mix(col, spc(sky_palette + 0.1, 0.8), sat(1.0 - length(col))) * 0.03;

    col *= 2.;

    return col;
}

vec3 sRGBToLinear(vec3 col) {
    return mix(pow((col + 0.055) / 1.055, vec3(2.4)), col / 12.92, lessThan(col, vec3(0.04045)));
}

vec3 linearTosRGB(vec3 col) {
    return mix(1.055 * pow(col, vec3(1.0 / 2.4)) - 0.055, col * 12.92,
               lessThan(col, vec3(0.0031308)));
}

float luminance(vec3 col) { return dot(col, vec3(0.2126729, 0.7151522, 0.0721750)); }
vec3 ReinhardExtLuma(vec3 col, const float w) {
    float l = luminance(col);
    float n = l * (1.0 + l / (w * w));
    float ln = n / (1.0 + l);
    return col * ln / l;
}

// https://cs.brown.edu/courses/cs129/results/proj5/valayshah/
// http://www.cs.nott.ac.uk/~pszqiu/webpages/Papers/icpr2006-hdri-camera.pdf
vec3 GlobalLogTonemap(vec3 col, const float ymin, const float ymax, const float t) {
    float y = luminance(col);
    float a = (ymax - ymin) * t;
    float lm = log(ymin + a);
    float yn = (log(y + a) - lm) / (log(ymax + a) - lm);
    return col * yn / y;
}

void main() {
    vec2 uv = (in_uv + -0.5) * vec2(pc.resolution.x / pc.resolution.y, 1.);
    vec2 m = pc.mouse * vec2(pc.resolution.x / pc.resolution.y, 1.);

    vec4 view_pos = pc.camera.cam.clip_to_world * vec4(in_uv - 0.5, 1., 1.);
    vec4 view_dir = pc.camera.cam.clip_to_world * vec4(in_uv - 0.5, 0., 1.);

    float t = pc.time;

    vec3 ro = vec3(0., 0., -3);
    vec3 rd = normalize(vec3(uv, 1));

    ro = view_pos.xyz / view_pos.w;
    rd = normalize(view_dir.xyz);
    ro = vec3(4.84, 6.94, 2.64);
    mat3 cam = get_cam(ro, ro + vec3(-.97, 3.7, 1.2));
    rd = normalize(cam * vec3(uv, 1.));

    vec3 col = vec3(0.96, 0.95, 0.9) * 0.2;

    vec3 sky_col = sky(rd);
    Hit hit = trace(ro, rd);
    if (hit.mat > 0) {
        vec3 pos = ro + rd * hit.t;
        vec3 normal = get_norm(pos, hit, t);

        vec3 rfl = reflect(rd, normal);
        rfl.y = abs(rfl.y);
        float fres = clamp((pow(1. - max(0.0, dot(-normal, rd)), 5.0)), 0.0, 1.0);
        col += sky(rfl) * fres * 0.9;

        vec3 waterCol = sat(sat(spc(0.46, 0.4)) * 0.05 * pow(min(pos.y + 0.5, 1.8), 4.0) *
                            length(sky_col) * (rd.z * 0.3 + 0.7));
        col += waterCol * 0.5;

        col = mix(col, sky_col, sat(1. - exp(-hit.t / MDIST * 2.5)));
    } else {
        col = sky_col;
    }
    // col *= 2.0;

    // col = col / (1.0 + col);
    // col = ACESFilm(col * 0.55);
    // col = Uncharted2Tonemap(col);
    // col = ReinhardExt(col, 5.5);
    col = ReinhardExtLuma(col, 2.5); // 5.5
    // col = col / (1.0 + luminance(col));
    // col = GlobalLogTonemap(col, 0.0, 3.5, 0.15);

    final_color = vec4(col, 1.);
}
