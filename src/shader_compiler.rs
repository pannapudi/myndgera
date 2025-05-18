use std::path::Path;

use crate::{SHADER_FOLDER, Watcher};
use anyhow::{Context, Result};
use shaderc::{CompilationArtifact, IncludeType, ShaderKind};

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions<'static>,
}

impl ShaderCompiler {
    pub fn new(watcher: &Watcher) -> Result<Self> {
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create shader compiler options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        if !cfg!(debug_assertions) {
            options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        }
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_generate_debug_info();

        let watcher_copy = watcher.clone();
        options.set_include_callback(move |name, include_type, source_file, _depth| {
            let path = match include_type {
                IncludeType::Relative => {
                    let mapping = watcher_copy.include_mapping.lock();
                    let source_file = mapping.get(Path::new(source_file)).unwrap();
                    let source_path = &source_file.iter().next().unwrap().path;
                    source_path.parent().unwrap().join(name)
                }
                IncludeType::Standard => Path::new(SHADER_FOLDER).join(name),
            };
            // TODO: recreate dependencies in case someone removes includes
            match std::fs::read_to_string(&path) {
                Ok(glsl_code) => {
                    let include_path = path.canonicalize().unwrap();
                    {
                        let mut watcher = watcher_copy.watcher.lock();
                        let _ = watcher.watch(&include_path, notify::RecursiveMode::NonRecursive);
                    }
                    {
                        let mut mapping = watcher_copy.include_mapping.lock();
                        let sources: Vec<_> =
                            mapping[Path::new(source_file)].iter().cloned().collect();
                        for source in sources {
                            mapping.entry(name.into()).or_default().insert(source);
                        }
                    }
                    Ok(shaderc::ResolvedInclude {
                        resolved_name: String::from(name),
                        content: glsl_code,
                    })
                }
                Err(err) => Err(format!(
                    "Failed to resolve include to {name} in {source_file} (was looking for {path:?}): {err}"
                )),
            }
        });

        Ok(Self {
            compiler: shaderc::Compiler::new().unwrap(),
            options,
        })
    }

    pub fn compile(&self, path: impl AsRef<Path>, kind: ShaderKind) -> Result<CompilationArtifact> {
        let source = std::fs::read_to_string(path.as_ref())?;
        Ok(self.compiler.compile_into_spirv(
            &source,
            kind,
            path.as_ref().file_name().and_then(|s| s.to_str()).unwrap(),
            // TODO: Don't use fixed entry point in the future
            "main",
            Some(&self.options),
        )?)
    }
}
