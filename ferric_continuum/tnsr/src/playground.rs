//! Shared helpers for the Bazel-backed `tnsr_demo` playground binary.

use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemoOptions {
    pub outputs: Vec<DemoOutput>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemoOutput {
    Dot(PathBuf),
    TraceJson(PathBuf),
}

impl DemoOptions {
    pub fn parse_from<I, S>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut args = args.into_iter().map(Into::into);
        let _program = args.next();
        let mut outputs = Vec::new();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--dot" => {
                    let path = args
                        .next()
                        .ok_or_else(|| "missing value for --dot".to_string())?;
                    outputs.push(DemoOutput::Dot(PathBuf::from(path)));
                }
                "--trace-json" => {
                    let path = args
                        .next()
                        .ok_or_else(|| "missing value for --trace-json".to_string())?;
                    outputs.push(DemoOutput::TraceJson(PathBuf::from(path)));
                }
                "--help" | "-h" => {
                    return Err(Self::usage());
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        if outputs.is_empty() {
            outputs.push(DemoOutput::Dot(PathBuf::from("/tmp/block.dot")));
        }

        Ok(Self { outputs })
    }

    pub fn usage() -> String {
        "\
Usage: tnsr_demo [--dot PATH] [--trace-json PATH]

Runs the tiny transformer autograd, checkpointing, scaling, memory, and
distributed-report playground. Generated graph/trace artifacts are written to
caller-selected output paths. With no flags, writes /tmp/block.dot.
"
        .to_string()
    }
}
