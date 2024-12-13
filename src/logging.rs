use eframe::egui::debug_text;
use log::{debug, error, log, info, warn};
use env_logger::{Builder, Target};
use chrono::Local;
use std::io::Write;
use std::fs::OpenOptions;
use std::fs::File;

pub type LevelFilter = log::LevelFilter;
pub type Level = log::Level;

pub struct LogConfig {
    pub log_file: String,
    pub log_level: LevelFilter,
}
impl LogConfig {
    pub fn new(log_file: String, log_level: LevelFilter) -> Self {
        Self { log_file, log_level }
    }
}

pub struct Logger {
    config: LogConfig,
}

impl Logger {
    pub fn new(config: LogConfig) -> Self {
        // Create the directory path if it doesn't exist
        if let Some(parent) = std::path::Path::new(&config.log_file).parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("Failed to create log directory: {}", e);
            });
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_file)
            .unwrap_or_else(|e| {
                eprintln!("Failed to open log file {}: {}", config.log_file, e);
                panic!("Could not open log file");
            });

        // Create a writer that writes to both stdout and file
        let multi_writer = MultiWriter {
            writers: vec![
                Box::new(std::io::stdout()),
                Box::new(file),
            ],
        };

        Builder::new()
            .format(|buf, record| {
                writeln!(buf, "{} [{}] {} {} - {}", 
                    Local::now().format("%Y-%m-%d %H:%M:%S"),
                    record.level(),
                    record.module_path().unwrap_or("unknown"),
                    record.file().unwrap_or("unknown"),
                    record.args())
            })
            .filter(None, config.log_level)
            .target(Target::Pipe(Box::new(multi_writer)))
            .init();

        Self { config}
    }

    pub fn log_event(
        &self,
        level: log::Level,
        event: &str,
        details: Option<&str>,
        error: Option<&dyn std::error::Error>,
    ) {
        match (details, error) {
            (Some(d), Some(e)) => log!(level, "{}: {} - {}", event, d, e),
            (Some(d), None) => log!(level, "{}: {}", event, d),
            (None, Some(e)) => log!(level, "{} - {}", event, e),
            (None, None) => log!(level, "{}", event),
        }
    }
}

// Add this struct above the init function
struct MultiWriter {
    writers: Vec<Box<dyn Write + Send + Sync>>,
}

impl Write for MultiWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for writer in &mut self.writers {
            writer.write_all(buf)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        for writer in &mut self.writers {
            writer.flush()?;
        }
        Ok(())
    }
}
