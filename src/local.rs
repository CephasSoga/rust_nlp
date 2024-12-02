//! # Local Model
//! 
//! A local model for sentiment analysis. 
//! 
//! Loads distilbert-base-uncased-finetuned-sst-2-english model from local files.
//! 
//! The model and all config files can be found at on the HuggingFace plateform.
//! 
//! Download the model and config files from the HuggingFace plateform [here](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) and put them in the `models` subfolder of the project.
//! 
//! Files to download:
//! - rust_model.ot
//! - config.json
//! - vocab.txt
//! 
//! The model and all config files are expected to be in the `models` folder.
//! 
//! This moduile uses the Rust-BERT crate to load the model.
//! 
//! ## Features
//!
//! - Sentiment analysis
//! 
//! ## Examples
//! 
//! ```
//! let model = ModelBuilder::new(ModelConfig::new("models/rust_model.ot", "models/config.json", "models/vocab.txt")).build_wrapper().unwrap();
//! let results = model.process(&["I love Rust!", "Rust is a great language!"]);
//! ```
//! 
//! ## References
//! 
//! - [Rust-BERT](https://github.com/guillaume-be/rust-bert)
//! - [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

#![allow(warnings)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::error::Error;
use std::path::PathBuf;
use std::clone::Clone;
use std::marker::Copy;

use serde::{Serialize, Deserialize};
use rust_bert::pipelines::sentiment::{Sentiment, SentimentModel, SentimentPolarity};

#[derive(Debug, Clone, Copy)]
/// Enum to represent the polarity of the content
/// 
/// # Fields
/// 
/// * `Positive` - The content is positive.
/// * `Negative` - The content is negative.
/// 

pub enum ContentPolarity {
    Positive,
    Negative,
}
impl ContentPolarity {
    /// Create a new `ContentPolarity` from a `SentimentPolarity`.
    /// 
    /// # Arguments
    /// 
    /// * `candidate` - The `SentimentPolarity` to convert.
    /// 
    /// # Returns
    /// 
    /// * `ContentPolarity::Positive` - The content is positive.
    /// * `ContentPolarity::Negative` - The content is negative.
    pub fn from_candidate(candidate: &SentimentPolarity) -> Self {
        match candidate {
            SentimentPolarity::Positive => ContentPolarity::Positive,
            SentimentPolarity::Negative => ContentPolarity::Negative,
        }
    }
}

#[derive(Clone, Copy)]
/// A candidate for the polarity of the content.
/// 
/// # Fields
/// 
/// * `label` - The polarity of the content.
/// * `score` - The score of the polarity.
pub struct ContentPolarityCandidate {
    pub label: ContentPolarity,
    pub score: f64,
}

/// A result of the parsing of the content.
/// 
/// # Fields
/// 
/// * `content` - The content to analyze.
/// * `polarity` - The polarity of the content.
/// 
/// # Examples
/// 
/// ```
/// let result = ParseResult { content: "I love Rust!".to_string(), polarity: ContentPolarityCandidate { label: ContentPolarity::Positive, score: 0.95 } };
/// ```
pub struct ParseResult {
    pub content: String,
    pub polarity: ContentPolarityCandidate,
}

/// A configuration for the model.
/// 
/// # Fields
/// 
/// * `model_path` - The path to the model.
/// * `config_path` - The path to the config.
/// * `vocab_path` - The path to the vocab.
/// 
/// # Examples
/// 
/// ```
/// let config = ModelConfig::new("models/rust_model.ot".to_string(), "models/config.json".to_string(), "models/vocab.txt".to_string());
/// config.load();
/// ```
pub struct ModelConfig {
    model_path: String,
    config_path: String,
    vocab_path: String,
}
impl ModelConfig {
    /// Create a new `ModelConfig`.
    /// 
    /// # Arguments
    /// 
    /// * `model_path` - The path to the model.
    /// * `config_path` - The path to the config.
    /// * `vocab_path` - The path to the vocab.
    pub fn new(model_path: String, config_path: String, vocab_path: String) -> Self {
        Self { model_path, config_path, vocab_path }
    }

    /// Loads the model.
    /// 
    /// # Returns
    /// 
    /// * `Ok(model)` - The loaded model.
    /// * `Err(e)` - An error if the model fails to load.
    pub fn load(&self) -> Result<SentimentModel, Box<dyn Error>> {
        let config = rust_bert::pipelines::sentiment::SentimentConfig {
            model_resource: PathBuf::from(&self.model_path).into(),
            config_resource: PathBuf::from(&self.config_path).into(),
            vocab_resource: PathBuf::from(&self.vocab_path).into(),
            ..Default::default()
        };
        Ok(SentimentModel::new(config)?)
    }
}

/// A wrapper for the model.
/// 
/// # Fields
/// 
/// * `model` - The loaded model.
/// 
/// # Examples
/// 
/// ```
/// let model = ModelWrapper::new(base_model);
/// model.process(&["I love Rust!", "Rust is a great language!"]);
/// ```
pub struct ModelWrapper {
    model: SentimentModel,
}
impl ModelWrapper {
    /// Create a new `ModelWrapper`.
    /// 
    /// # Arguments
    /// 
    /// * `model` - The loaded model.
    /// 
    /// # Returns
    /// 
    /// * `ModelWrapper` - The new model wrapper.
    pub fn new(model: SentimentModel) -> Self {
        Self { model }
    }

    /// Processes the text.
    /// 
    /// # Arguments
    /// 
    /// * `text` - The text to process.
    /// 
    /// # Returns
    /// 
    /// * `Vec<Sentiment>` - The processed text.
    pub fn process(&self, text: &[&str]) -> Vec<Sentiment> {
        let output = self.model.predict(text);
        output
    }

    /// Parses the output of the model.
    /// 
    /// # Arguments
    /// 
    /// * `texts` - The texts to parse.
    /// * `output` - The output of the model.
    /// 
    /// # Returns
    /// 
    /// * `Vec<ParseResult>` - The parsed results.
    pub fn parse(&self, texts: &[&str], output: Vec<Sentiment>) -> Vec<ParseResult> {
        let mut results = Vec::new();
        for (i, sentiment) in output.iter().enumerate() {
            results.push(ParseResult {
                content: texts[i].to_string(),
                polarity: ContentPolarityCandidate { 
                    label: ContentPolarity::from_candidate(&sentiment.polarity),
                    score: sentiment.score
                },
            });
        }
        results
    }
}

/// A builder for the model.
/// 
/// # Fields
/// 
/// * `model_config` - The configuration for the model.
/// 
/// # Examples
/// 
/// ```
/// let builder = ModelBuilder::new(config);
/// builder.build_wrapper();
/// ```
pub struct ModelBuilder {
    model_config: ModelConfig,
}
impl ModelBuilder {
    /// Create a new `ModelBuilder`.
    /// 
    /// # Arguments
    /// 
    /// * `model_config` - The configuration for the model.
    /// 
    /// # Returns
    /// 
    /// * `ModelBuilder` - The new model builder.
    pub fn new(model_config: ModelConfig) -> Self {
        Self { model_config }
   
    }

    /// Builds the model wrapper.
    /// 
    /// # Returns
    /// 
    /// * `Ok(model)` - The built model wrapper.
    /// * `Err(e)` - An error if the model wrapper fails to build.
    pub fn build_wrapper(&self) -> Result<ModelWrapper, Box<dyn Error>> {
        let model_config = ModelConfig::new(    
            "models/rust_model.ot".to_string(),
            "models/config.json".to_string(),
            "models/vocab.txt".to_string()
        );
        let base_model = model_config.load()?;
        let model = ModelWrapper::new(base_model);
        Ok(model)
    }
}
