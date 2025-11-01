use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use notify::{
    EventKind, RecommendedWatcher,
    event::{AccessKind, AccessMode},
};
use notify_debouncer_full::{DebounceEventResult, RecommendedCache};
use winit::event_loop::EventLoopProxy;

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use crate::{ShaderSource, UserEvent};

use parking_lot::Mutex;

#[derive(Clone)]
pub struct Watcher {
    pub watcher: Arc<Mutex<notify_debouncer_full::Debouncer<RecommendedWatcher, RecommendedCache>>>,
    pub include_mapping: Arc<Mutex<AHashMap<PathBuf, AHashSet<ShaderSource>>>>,
}

impl Watcher {
    pub fn new(proxy: EventLoopProxy<UserEvent>) -> Result<Self> {
        let watcher = notify_debouncer_full::new_debouncer(
            Duration::from_millis(150),
            None,
            watch_callback(proxy),
        )?;

        Ok(Self {
            watcher: Arc::new(Mutex::new(watcher)),
            include_mapping: Arc::new(Mutex::new(AHashMap::new())),
        })
    }

    pub fn unwatch_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let mut watcher = self.watcher.lock();
        watcher.unwatch(path.as_ref())?;
        Ok(())
    }

    pub fn watch_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let mut watcher = self.watcher.lock();
        let mut path = path.as_ref();
        if path.is_file() {
            path = path.parent().unwrap();
        }
        watcher.watch(path, notify::RecursiveMode::Recursive)?;
        Ok(())
    }
}

fn watch_callback(proxy: EventLoopProxy<UserEvent>) -> impl FnMut(DebounceEventResult) {
    move |event| match event {
        Ok(events) => {
            for path in events
                .into_iter()
                .filter(|e| {
                    matches!(
                        e.event.kind,
                        EventKind::Modify(_)
                            | EventKind::Access(AccessKind::Close(AccessMode::Write))
                    )
                })
                .filter_map(|event| event.event.paths.into_iter().next())
            {
                if path.extension() == Some(OsStr::new("glsl"))
                    || path.extension() == Some(OsStr::new("frag"))
                    || path.extension() == Some(OsStr::new("vert"))
                    || path.extension() == Some(OsStr::new("comp"))
                {
                    let _ = proxy
                        .send_event(UserEvent::Glsl {
                            path: path.canonicalize().unwrap(),
                        })
                        .map_err(|err| tracing::error!("Event Loop has been dropped: {err}"));
                }
            }
        }
        Err(errors) => tracing::error!("File watcher error: {errors:?}"),
    }
}
