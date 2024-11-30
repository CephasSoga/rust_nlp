#![allow(warnings)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::error::Error;
use std::path::PathBuf;
use std::clone::Clone;
use std::marker::Copy;
use std::collections::{HashMap, HashSet};
use std::cmp::Reverse;

use regex::Regex;
use serde::{Serialize, Deserialize};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use rust_bert::pipelines::common::ModelType;
use rust_bert::resources::RemoteResource;
use tch::Device;
use ndarray::{Array2, ArrayView2};
use fastrand;

const STOPWORDS_PATH: &str = "data/stopwords.txt";

#[derive(Debug, Clone, Deserialize, Serialize)]
/// A document with an ID, text, vector, and bag of words
/// 
/// # Fields
/// * `id`: A unique identifier for the document
/// * `text`: The text of the document
/// * `vector`: The vector representation of the document
/// * `bag_of_words`: The bag of words representation of the document
pub struct Document {
    pub id: String,
    pub text: String,
    pub vector: Option<Vec<f32>>,
    pub bag_of_words: Option<HashMap<String, usize>>,
}
impl Document {
    /// Generate a random ID for a document
    /// 
    /// # Returns
    /// A random ID as a string
    pub fn generate_id() -> String {
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        (0..8)
            .map(|_| {
                let idx = fastrand::usize(..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    /// Create a new document
    /// 
    /// # Arguments
    /// * `text`: The text of the document
    /// 
    /// # Returns
    /// A new document
    pub fn new(text: String) -> Self {
        Self { 
            id: Self::generate_id(),  // Use string ID instead of usize
            text, 
            vector: None, 
            bag_of_words: None 
        }
    }

    /// Embed the document using a preprocessor and a topic model
    /// 
    /// # Arguments
    /// * `processor`: The preprocessor to use
    /// * `model`: The topic model to use
    /// 
    /// # Returns
    /// The vector representation of the document
    /// 
    /// # Examples
    /// ```
    /// let doc = Document::new("Rust is a great programming language!");
    /// let processor = Preprocessor::new();
    /// let model = TopicModel::new();
    /// doc.embed(&processor, &model);
    /// ```
    pub fn embed(&mut self, processor: &Preprocessor, model: &TopicModel) {
        let processed_tokens = processor.preprocess(&self.text);
        let vector = model.compute_tfidf(&processed_tokens);
        self.vector = Some(vector);
    }

    /// Compute the bag of words for a document
    /// 
    /// # Arguments
    /// * `processor`: The preprocessor to use
    /// * `model`: The topic model to use
    /// 
    /// # Examples
    /// ```
    /// let doc = Document::new("Rust is a great programming language!");
    /// let processor = Preprocessor::new();
    /// let model = TopicModel::new();
    /// doc.bag_of_words(&processor, &model);
    /// ```
    pub fn bag_of_words(&mut self, processor: &Preprocessor, model: &TopicModel) {
        let processed_tokens = processor.preprocess(&self.text);
        self.bag_of_words = Some(model.bag_of_words(&processed_tokens));
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
/// A cluster with an ID, documents, topics, and size
/// 
/// # Fields
/// * `id`: The ID of the cluster
/// * `documents`: The documents in the cluster
/// * `topics`: The topics in the cluster
/// * `size`: The size of the cluster
pub struct Cluster {
    pub id: usize,
    pub documents: Vec<Document>,
    pub topics: Vec<String>,
    pub size: usize,
}
impl Cluster {
    /// Create a new cluster
    /// 
    /// # Arguments
    /// * `id`: The ID of the cluster
    /// * `documents`: The documents in the cluster
    /// * `topics`: The topics in the cluster
    /// * `size`: The size of the cluster
    pub fn new(id: usize, documents: Option<Vec<Document>>, topic: Option<String>, size: Option<usize>) -> Self {
        Self { 
            id, 
            documents: documents.unwrap_or(vec![]), 
            topics: vec![],
            size: size.unwrap_or(0) 
        }
    }

    /// Update the cluster with documents and topics
    /// 
    /// # Arguments
    /// * `documents`: The documents to add to the cluster
    /// * `topic`: The topic to add to the cluster
    /// 
    /// # Examples
    /// ```
    /// let cluster = Cluster::new(0, None, None, None);
    /// cluster.update(Some(vec![Document::new("Rust is a great programming language!")]), Some("Rust"));
    /// ```
    pub fn update(&mut self, documents: Option<Vec<Document>>, topic: Option<String>) {
        self.documents.extend(documents.unwrap_or(vec![]));
        self.topics.push(topic.unwrap_or(String::new()));
        self.size = self.documents.len();
    }
    
    /// Clear the cluster
    /// 
    /// # Examples
    /// ```
    /// let cluster = Cluster::new(0, None, None, None);
    /// cluster.clear();
    /// ```
    pub fn clear(&mut self) {
        self.topics.clear();
        self.documents.clear();
        self.size = 0;
    }

    /// Destroy the cluster
    /// 
    /// # Examples
    /// ```
    /// let cluster = Cluster::new(0, None, None, None);
    /// cluster.destroy();
    /// ```
    pub fn destroy(self) {
        drop(self);
    }

    /// Convert the cluster to a JSON string
    /// 
    /// # Returns
    /// The cluster as a JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

/// A preprocessor for text
///     
/// # Fields
/// * `stopwords`: The stopwords in the preprocessor
pub struct Preprocessor {
    stopwords: Vec<String>,
}

impl Preprocessor {
    /// Create a new preprocessor
    pub fn new() -> Self {
        Self { stopwords: vec![] }
    }

    /// Read the stopwords from a file
    /// 
    /// # Arguments
    /// * `txt_file_path`: The path to the stopwords file
    /// 
    /// # Returns
    /// The stopwords as a vector of strings
    /// 
    /// # Examples
    /// ```
    /// let mut processor = Preprocessor::new();
    /// processor.read_stopwords(None).unwrap();
    /// ```
    pub fn read_stopwords(&mut self, txt_file_path: Option<&str>) -> io::Result<Vec<String>> {
        let path = txt_file_path.unwrap_or(STOPWORDS_PATH);
        let file = File::open(&path)?;
        let stopwords: Vec<String> = io::BufReader::new(file)
            .lines()
            .map(|line| line.unwrap().trim().to_string()) // Trim and convert to String
            .collect();
        self.stopwords.extend(stopwords.clone());
        Ok(stopwords)
    }

    /// Preprocesses a given text
    /// 
    /// # Arguments
    /// * `text`: The text to preprocess
    /// 
    /// # Returns
    /// The preprocessed text as a vector of strings
    /// 
    /// # Examples
    /// ```
    /// let mut processor = Preprocessor::new();
    /// processor.read_stopwords(None).unwrap();
    /// let processed_text = processor.preprocess("Rust is a great programming language!");
    /// ```
    pub fn preprocess(&self, text: &str) -> Vec<String> {
        let lowercased = text.to_lowercase();

        let no_punctuation = Regex::new(r"[^\w\s]")
            .unwrap()
            .replace_all(&lowercased, "");

        let tokens: Vec<String> = no_punctuation
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let filtered: Vec<String> = tokens
            .into_iter()
            .filter(|token| !self.stopwords.contains(token))
            .collect();

        filtered
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
/// A configuration for a topic model
/// 
/// # Fields
/// * `texts`: The texts to fit the topic model to
/// * `n_clusters`: The number of clusters to fit the topic model to
/// * `k_topics`: The number of topics to fit the topic model to
pub struct TopicModelConfig {
    pub texts: Vec<String>,
    pub n_clusters: usize,
    pub k_topics: usize,
 }
 impl TopicModelConfig {
    /// Create a new topic model configuration
    /// 
    /// # Arguments
    /// * `texts`: The texts to fit the topic model to
    /// * `n_clusters`: The number of clusters to fit the topic model to
    /// * `k_topics`: The number of topics to fit the topic model to
    pub fn new(texts: Vec<String>, n_clusters: Option<usize>, k_topics: Option<usize>) -> Self {
        let n_clusters = n_clusters.unwrap_or(10);
        let k_topics = k_topics.unwrap_or(5);
        Self { texts, n_clusters, k_topics }
    }
 }


/// A topic model
/// 
/// # Fields
/// * `preprocessor`: The preprocessor for the topic model
pub struct TopicModel {
    preprocessor: Preprocessor,
}

impl TopicModel {
    /// Create a new topic model
    pub fn new() -> Self {
        Self { 
            preprocessor: {
                let mut pr = Preprocessor::new();
                pr.read_stopwords(None).unwrap();
                pr
            }
        }
    }

    /// Compute the bag of words for a set of documents
    /// 
    /// # Arguments
    /// * `documents`: The documents to compute the bag of words for
    /// 
    /// # Returns
    /// The bag of words as a hash map of strings to usizes
    /// 
    /// # Examples
    /// ```
    /// let mut model = TopicModel::new();
    /// let bag_of_words = model.bag_of_words(&["Rust is a great programming language!", "Rust is a great language!"]);
    /// ```
    pub fn bag_of_words(&self, documents: &[String]) -> HashMap<String, usize> {
        let mut word_count = HashMap::new();
        for doc in documents {
            for word in doc.split_whitespace() {
                let word = word.to_lowercase();
                *word_count.entry(word).or_insert(0) += 1;
            }
        }
        word_count
    }

    /// Compute the TF-IDF for a set of documents
    /// 
    /// # Arguments
    /// * `documents`: The documents to compute the TF-IDF for
    /// 
    /// # Returns
    /// The TF-IDF as a vector of f32s
    /// 
    /// # Examples
    /// ```
    /// let mut model = TopicModel::new();
    /// let tfidf = model.compute_tfidf(&["Rust is a great programming language!", "Rust is a great language!"]);
    /// ```
    pub fn compute_tfidf(&self, documents: &[String]) -> Vec<f32> {
        // Initialize vectors
        let mut term_frequencies: Vec<f32> = Vec::new();
        let mut document_frequencies = HashMap::new();
        let num_documents = documents.len() as f32;
        let mut all_terms = HashSet::new();

        // First pass: collect all unique terms and document frequencies
        for doc in documents {
            let mut seen_terms = HashSet::new();
            for term in doc.split_whitespace() {
                let term = term.to_lowercase();
                all_terms.insert(term.clone());
                if seen_terms.insert(term.clone()) {
                    *document_frequencies.entry(term).or_insert(0.0) += 1.0;
                }
            }
        }

        // Convert terms to vector
        let terms: Vec<String> = all_terms.into_iter().collect();
        
        // Compute TF-IDF scores for each document
        let mut tfidf_scores = Vec::new();
        for doc in documents {
            let mut term_freq = vec![0.0; terms.len()];
            let words: Vec<&str> = doc.split_whitespace().collect();
            let doc_len = words.len() as f32;

            // Calculate term frequencies
            for word in words {
                let word = word.to_lowercase();
                if let Some(idx) = terms.iter().position(|t| t == &word) {
                    term_freq[idx] += 1.0 / doc_len;
                }
            }

            // Calculate TF-IDF
            for (idx, tf) in term_freq.iter_mut().enumerate() {
                let df = document_frequencies.get(&terms[idx]).unwrap();
                let idf = (num_documents / df).ln();
                *tf *= idf;
            }

            tfidf_scores.extend(term_freq);
        }

        tfidf_scores
    }


    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        let magnitude_b = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        dot_product / (magnitude_a * magnitude_b)
    }

    /// Cluster a set of documents
    /// 
    /// # Arguments
    /// * `documents`: The documents to cluster
    /// * `n_clusters`: The number of clusters to create
    /// 
    /// # Returns
    /// The clusters as a vector of clusters
    /// 
    /// # Examples
    /// ```
    /// let mut model = TopicModel::new();
    /// let clusters = model.cluster(&[Document::new("Rust is a great programming language!")], Some(3));
    /// ```
    pub fn cluster(&self, documents: &[Document], n_clusters: Option<usize>) -> Vec<Cluster> {
        // Initialize clusters
        let n_clusters = n_clusters.unwrap_or(3);
        let mut clusters: Vec<Cluster> = (0..n_clusters)
            .map(|id| Cluster::new(id, None, None, None))
            .collect();

        // Ensure documents have vectors
        let doc_vectors: Vec<&Vec<f32>> = documents.iter()
            .filter_map(|doc| doc.vector.as_ref())
            .collect();

        // If no vectors, return empty clusters
        if doc_vectors.is_empty() {
            return clusters;
        }

        // Get the dimension of the vectors
        let vector_dim = doc_vectors[0].len();
        
        // Initialize centroids randomly by picking random documents
        let mut centroids: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| doc_vectors[fastrand::usize(..doc_vectors.len())].clone())
            .collect();

        // Initialize previous centroids
        let max_iterations = 100;
        let mut prev_centroids: Vec<Vec<f32>>;

        // K-means iteration
        for _ in 0..max_iterations {
            prev_centroids = centroids.clone();

            // Clear existing clusters
            for cluster in clusters.iter_mut() {
                cluster.clear();
            }

            // Assign documents to nearest centroid
            for (idx, doc) in documents.iter().enumerate() {
                if let Some(doc_vec) = &doc.vector {
                    let mut min_dist = f64::MAX;
                    let mut closest_cluster = 0;

                    // Find the closest centroid
                    for (centroid_idx, centroid) in centroids.iter().enumerate() {
                        let dist = doc_vec.iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>();

                        if (dist as f64) < min_dist {
                            min_dist = dist as f64;
                            closest_cluster = centroid_idx;
                        }
                    }

                    clusters[closest_cluster].documents.push(doc.clone());
                }
            }

            // Update centroids
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.documents.is_empty() {
                    let mut new_centroid = vec![0.0; vector_dim];
                    
                    for doc in &cluster.documents {
                        if let Some(vec) = &doc.vector {
                            for (j, val) in vec.iter().enumerate() {
                                if j < new_centroid.len() {
                                    new_centroid[j] += val;
                                }
                            }
                        }
                    }

                    for val in new_centroid.iter_mut() {
                        *val /= cluster.documents.len() as f32;
                    }

                    centroids[i] = new_centroid;
                }
            }

            // Check convergence
            let converged = centroids.iter().zip(prev_centroids.iter())
                .all(|(new, old)| {
                    new.iter().zip(old.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-4)
                });

            if converged {
                break;
            }
        }

        // Update cluster sizes
        for cluster in clusters.iter_mut() {
            cluster.size = cluster.documents.len();
        }

        clusters
    }

    /// Fit a topic model to a set of documents
    /// 
    /// # Arguments
    /// * `config`: The configuration for the topic model
    /// 
    /// # Returns
    /// The clusters as a vector of clusters
    /// 
    /// # Examples
    /// ```
    /// let mut model = TopicModel::new();
    /// let clusters = model.fit(TopicModelConfig::new(vec!["Rust is a great programming language!", "Rust is a great language!"], Some(3), Some(5)));
    /// ```
    pub fn fit(&self, config: TopicModelConfig) -> Vec<Cluster> {
        // Extract configuration
        let TopicModelConfig { texts, n_clusters, k_topics } = config;

        // Create documents
        let mut documents: Vec<Document> = texts.iter()
            .map(|text| Document::new(text.to_string()))
            .collect();

        // Embed and compute bag of words
        for doc in &mut documents {
            doc.embed(&self.preprocessor, self);
            doc.bag_of_words(&self.preprocessor, self);
        }

        // Cluster the documents
        let mut clusters = self.cluster(&documents, Some(n_clusters));

        // Compute the top k topics for each cluster
        for cluster in &mut clusters {
            cluster.topics = self.cluster_top_k_topics(cluster, k_topics);
        }
        
        clusters
    }

    /// Compute the top k topics for a cluster
    /// 
    /// # Arguments
    /// * `cluster`: The cluster to compute the top k topics for
    /// * `k`: The number of topics to compute
    /// 
    /// # Returns
    /// The top k topics as a vector of strings
    /// 
    /// # Examples
    /// ```
    /// let mut model = TopicModel::new();
    /// let topics = model.cluster_top_k_topics(&Cluster::new(0, None, None, None), 5);
    /// ```
    pub fn cluster_top_k_topics(&self, cluster: &Cluster, k: usize) -> Vec<String> {
        let mut merged_counts: HashMap<String, usize> = HashMap::new();
        
        // Merge word counts from all documents
        for doc in &cluster.documents {
            if let Some(bag_of_words) = &doc.bag_of_words {
                for (word, count) in bag_of_words {
                    *merged_counts.entry(word.clone()).or_insert(0) += count;
                }
            }
        }

        // Convert to vec, sort, and take top k
        let mut words: Vec<_> = merged_counts.into_iter().collect();
        words.sort_by_key(|(_, count)| Reverse(*count));
        words.into_iter()
            .take(k)
            .map(|(word, _)| word)
            .collect()
    }
}



/// Example usage
fn main() {
    // Create the preprocessor
    let preprocessor = Preprocessor::new();

    // Example text
    let text = "Rust is a great programming language!";

    // Preprocess the text
    let processed_tokens = preprocessor.preprocess(text);

    // Print the processed tokens
    println!("{:?}", processed_tokens);

    let docs = vec![
        String::from("I love Rust programming."),
        String::from("Rust is great for system programming."),
    ];

    let topic_model = TopicModel::new();
    let tfidf = topic_model.compute_tfidf(&docs);
    println!("{:?}", tfidf);
}
