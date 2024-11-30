#![allow(warnings)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

mod local;
mod topic;
mod pattern;

use std::env;
use std::clone::Clone;
use std::error::Error;
use std::str::FromStr;
use std::marker::Copy;
use std::time::Instant;


use tokio;
use reqwest;
use regex::Regex;
use dotenv::dotenv;
use serde::{Serialize, Deserialize};
use rust_bert::pipelines::sentiment::{SentimentModel, Sentiment, SentimentPolarity};

use local::{ModelWrapper, ModelConfig, ContentPolarity, ModelBuilder};
use topic::{TopicModel, TopicModelConfig};

#[derive(Debug, Clone, Copy)]
/// The polarity of a piece of content.
/// 
/// # Variants
/// 
/// * `Positive` - The content is positive.
/// * `Neutral` - The content is neutral.
/// * `Negative` - The content is negative.
pub enum Polarity {
    Positive,
    Neutral,
    Negative,
}
impl Polarity {
    /// Create a `Polarity` from a `PolarityCandidate`.
    /// 
    /// # Arguments
    /// 
    /// * `candidate` - The candidate to convert to a `Polarity`.
    /// 
    /// # Returns
    /// 
    /// * `Ok(Polarity)` - The `Polarity` that corresponds to the candidate.
    /// * `Err(&'static str)` - The error message if the candidate is invalid.
    fn from_candidate (candidate: PolarityCandidate) -> Result<Self, &'static str> {
        // Convert the label to a `Polarity`
        match candidate.label.as_str() {
            "Positive"|"positive" => Ok(Polarity::Positive),
            "Neutral"|"neutral" => Ok(Polarity::Neutral),
            "Negative"|"negative" => Ok(Polarity::Negative),
            _ => Err("Invalid polarity argument"),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
/// A candidate for the polarity of a piece of content.
/// 
/// # Fields
/// 
/// * `label` - The label of the candidate.
/// * `score` - The score of the candidate.
struct PolarityCandidate {
    label: String,
    score: f32,
}
impl PolarityCandidate {
    /// Compare two `PolarityCandidate`s and return the one with the higher score.
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other `PolarityCandidate` to compare to.
    /// 
    /// # Returns
    /// 
    /// * `Self` - The `PolarityCandidate` with the higher score.
    fn compare(self, other: Self) -> Self {
        if self.score > other.score { self } else { other }
    }
}

#[derive(Debug, Clone, Copy)]
/// The subjectivity of a piece of content.
/// 
/// # Variants
/// 
/// * `Subjective` - The content is subjective.
/// * `Objective` - The content is objective.
pub enum Subjectivity {
    Subjective,
    Objective,
}
impl Subjectivity {
    /// Compute the subjectivity of a piece of content.
    /// 
    /// # Arguments
    /// 
    /// * `text` - The content to compute the subjectivity of.
    /// 
    /// # Returns
    /// 
    /// * `(Subjectivity, f32)` - The subjectivity and the score.
    fn compute(text: &str) -> (Self, f32) {
        let pattern = r"(?i)\b(I|you|me|my|mine|yours|yourself|himself|herself|itself|ourselves|yourselves|themselves)\b[,.!?]?";
        let re = Regex::new(pattern)
            .expect("Failed to compile regex pattern");

        // Find and count matches for subjective content
        let subjective_count = re.captures_iter(text).count() as f32;

        // Split text into sentences
        let total_sentences = Regex::new(r"[.!?]")
            .expect("Failed to compile sentence-splitting regex")
            .split(text)
            .filter(|sentence| !sentence.trim().is_empty())
            .count() as f32;

        // Compute the subjectivity score
        let subjectivity_score = if total_sentences > 0.0 {
            subjective_count / total_sentences
        } else {
            0.0 // Avoid division by zero
        };

        // Determine subjectivity type
        let subjectivity = if subjectivity_score < 0.5 {
            Subjectivity::Objective
        } else {
            Subjectivity::Subjective
        };

        (subjectivity, subjectivity_score)
    }
}


#[derive(Debug, Deserialize)]
/// The response from the Hugging Face API.
/// 
/// # Fields
/// 
/// * `results` - The results of the API call.
struct HuggingFaceApiResponse {
    results: Vec<PolarityCandidate>,
}

#[derive(Serialize, Deserialize)]
/// The configuration for the Hugging Face API client.
/// 
/// # Fields
/// 
/// * `api_url` - The URL of the API.
/// * `api_token` - The token for the API.
struct HuggingFaceApiClientConfig {
    api_url: String,
    api_token: String,
}
impl HuggingFaceApiClientConfig {
    /// Create a new `HuggingFaceApiClientConfig`.
    /// 
    /// # Arguments
    /// 
    /// * `api_url` - The URL of the API.
    /// * `api_token` - The token for the API.
    /// 
    /// # Returns
    /// 
    /// * `Self` - The `HuggingFaceApiClientConfig`.
    fn new(api_url: Option<String>, api_token: Option<String>) -> Self {

        Self {
            // Set the API URL from the environment variable if it exists, otherwise use the default
            api_url: api_url.unwrap_or_else(|| env::var("HUGGINGFACE_API_URL").expect("HUGGINGFACE_API_URL must be set")),
            // Set the API token from the environment variable if it exists, otherwise use the default
            api_token: api_token.unwrap_or_else(|| env::var("HUGGINGFACE_API_TOKEN").expect("HUGGINGFACE_API_TOKEN must be set")),
        }
    }
}

/// The client for the Hugging Face API.
/// 
/// # Fields
/// 
/// * `config` - The configuration for the API client.
/// * `client` - The HTTP client.
struct HuggingFaceApiClient {
    config: HuggingFaceApiClientConfig,
    client: reqwest::Client,
}
impl HuggingFaceApiClient {
    /// Create a new `HuggingFaceApiClient`.
    /// 
    /// # Arguments
    /// 
    /// * `config` - The configuration for the API client.
    /// * `client` - The HTTP client.
    /// 
    /// # Returns
    /// 
    /// * `Self` - The `HuggingFaceApiClient`.
    fn new(config: HuggingFaceApiClientConfig, client: reqwest::Client) -> Self {
        Self { config, client}
    }

    /// Asynchronously compute the polarity of a piece of content.
    /// 
    /// # Arguments
    /// 
    /// * `content` - The content to compute the polarity of.
    /// 
    /// # Returns
    /// 
    /// * `Result<(Polarity, f32), Box<dyn Error>>` - The polarity and the score.
    async fn compute_polarity(&self, content: &str) -> Result<(Polarity, f32), Box<dyn Error>> {
        let response = self.client.post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", &self.config.api_token))
            .json(&serde_json::json!({ "inputs": content }))
            .send()
            .await?;

        // Parse the response
        let json_probs: HuggingFaceApiResponse = response.json().await?;

        // Get the polarity candidate with the highest score
        let result = if json_probs.results[0].score > json_probs.results[1].score {
            &json_probs.results[0]
        } else {
            &json_probs.results[1]
        };
        // Convert the polarity candidate to a `Polarity`
        let polarity = Polarity::from_candidate(result.clone()).unwrap();
        // Get the score of the polarity candidate
        let score = result.score;

        Ok((polarity, score))
    }
}

#[derive(Debug)]
enum PolarityResult {
    Api(Polarity, f32),
    Local(ContentPolarity, f64),
}

#[derive(Debug)]
/// A wrapper for the sentiment of a piece of content.
/// 
/// # Fields
/// 
/// * `content` - The content to analyze.
/// * `polarity` - The polarity of the content.
/// * `subjectivity` - The subjectivity of the content.
struct SentimentWrapper {
    content: String,
    polarity: PolarityResult,
    subjectivity: (Subjectivity, f32),
}
impl SentimentWrapper {
    /// Create a new `SentimentWrapper`.
    /// 
    /// # Arguments
    /// 
    /// * `content` - The content to analyze.
    /// * `polarity` - The polarity of the content.
    /// * `subjectivity` - The subjectivity of the content.
    fn new(content: String, polarity: Option<PolarityResult>, subjectivity: Option<(Subjectivity, f32)>) -> Self {
        let polarity = polarity.unwrap_or(PolarityResult::Api(Polarity::Neutral, 0.0));
        let subjectivity = subjectivity.unwrap_or((Subjectivity::Objective, 0.0));
        Self { content, polarity, subjectivity }
    }

    /// Compute the polarity of a piece of content using the local model.
    /// 
    /// # Arguments
    /// 
    /// * `model` - The model to use to compute the polarity.
    /// * `content` - The content to analyze.
    /// 
    /// # Returns
    /// 
    /// * `Result<(ContentPolarity, f64), Box<dyn Error>>` - The polarity and the score.
    fn compupte_polarity_from_local(model: &ModelWrapper, content: &str) -> Result<( ContentPolarity, f64), Box<dyn Error>> {
        let output = model.process(&[content]);
        let result = model.parse(&[content], output);
        Ok((result[0].polarity.label, result[0].polarity.score))
    }

    /// Asynchronously analyze the sentiment of a piece of content.
    /// 
    /// # Arguments 
    /// 
    /// * `model` - The model to use to analyze the sentiment.
    /// 
    /// # Returns
    /// 
    /// * `Result<(), Box<dyn Error>>` - The result of the analysis.
    async fn analyze(&mut self, model: &ModelWrapper) -> Result<(), Box<dyn Error>> {
        let (subjectivity, subjectivity_score) = Subjectivity::compute(&self.content);
        self.subjectivity = (subjectivity, subjectivity_score);

        // Initialize the Hugging Face API client
        let req_client = reqwest::Client::new();
        // Initialize the Hugging Face API client
        // ### Uncomment the following line to use the Hugging Face API ###
        // let hf_client = HuggingFaceApiClient::new(
        //     HuggingFaceApiClientConfig::new(None, None), req_client
        // );
        // let (polarity, polarity_score) = hf_client.compute_polarity(&self.content).await?;
        // self.polarity = (polarity, polarity_score);  

        // Compute the polarity of the content using the local model
        // ### Uncomment the following line to use the local model ###
        let (polarity, polarity_score) = Self::compupte_polarity_from_local(&model, &self.content)?;
        self.polarity = PolarityResult::Local(polarity, polarity_score);
        Ok(())
    }
}


/// Macro to time the execution of a block of code
#[macro_export]
macro_rules! time_it {
    ($label:expr, $block:expr) => {{
        let start = Instant::now();
        let result = $block;
        let duration = start.elapsed();
        println!("{} took {:.8?}", $label, duration);
        result
    }};
}

#[tokio::main]
/// The main function.
async fn main() {
        // Load environment variables from .env file
        dotenv().ok();
        // Initialize the model config arguments
        let model_path = "models/rust_model.ot".to_string();
        let config_path = "models/config.json".to_string();
        let vocab_path = "models/vocab.txt".to_string();
        // build the model wrapper with the builder
        let model = ModelBuilder::new(
            ModelConfig::new(model_path, config_path, vocab_path)
        )
        .build_wrapper()
        .unwrap();

    let texts = vec![
        String::from("I am a happy person. I am learning Rust. It is fun and challenging. This is a message to me"),
        String::from("Graphical models are a powerful tool for modeling complex systems. They are used in a wide range of applications, from natural language processing to computer vision."),
        String::from("The future of AI is exciting. I am looking forward to see what new applications will emerge in the future."),
    ];
    // Time the sentiment analysis
    time_it!("Sentiment analysis", {
        // Initialize the sentiment wrapper
        let mut sentiment = SentimentWrapper::new(texts[0].clone(), None, None);
        // Analyze the sentiment of the content
        sentiment.analyze(&model).await.unwrap();
        // Print the sentiment
        println!("{:?}", sentiment);

    });

    // Time the topic modeling
    time_it!("Topic modeling", {
        // Initialize the topic model
        let topic_model = TopicModel::new();
        // Fit the topic model
        let clusters = topic_model.fit(TopicModelConfig::new(texts, None, None));
        // Print the clusters
        println!("{:?}", clusters);
    });
}
