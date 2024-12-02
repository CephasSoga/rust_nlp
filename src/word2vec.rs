//! Word2Vec model implementation.
//! 
//! Based on the paper "Distributed Representations of Words and Phrases and their Compositionality" by Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean.
//! 
//! https://arxiv.org/abs/1310.4546
//! 
//! 
//! # Single-threaded Word2Vec
//! It supports key functionalities such as reading a corpus from a file, 
//! training using the Skip-Gram with Negative Sampling approach, 
//! and saving/loading the model in JSON format for persistence.
//!
//! ## Key Features
//! 
//! ### Initialization:
//! 
//! The Word2Vec::new method initializes random embeddings for input and output layers.
//! Uses fastrand for efficient random number generation.
//! 
//! ### Training:
//! 
//! Implements Skip-Gram with Negative Sampling for efficient training on large vocabularies.
//! Context window size and number of negative samples are configurable.
//! 
//! ### Persistence:
//! 
//! The save and load methods allow saving/loading the model to/from a JSON file 
//! using serde_json.
//! 
//! ### Corpus Handling:
//! 
//! The read_corpus method reads and preprocesses the input text from a file.
//! 
//! ### Update Mechanism:
//! 
//! Gradient descent is used to update word embeddings during training.
//! 
//! 
//! # Multi-threaded Word2Vec
//! 
//! This MultithreadWord2Vec structure introduces thread safety and parallel processing, 
//! which can improve performance when training the Word2Vec model on a large corpus. 
//! By using Arc<Mutex<_>> for the shared input_vectors and output_vectors, 
//! you allow multiple threads to safely update these shared resources during training. 
//! However, this approach has trade-offs and potential bottlenecks. 
//! Let's examine its benefits and drawbacks compared to the original method:
//! 
//! ## Benefits
//! Parallelism for Faster Training:
//! 
//! The par_iter() from Rayon enables parallel processing of the corpus, 
//! which can speed up training significantly if your system has multiple cores.
//! 
//! ### Thread Safety with Arc<Mutex<_>>:
//! 
//! The Arc<Mutex<_>> ensures that updates to shared data structures 
//! (e.g., input and output vectors) are synchronized, preventing data races.
//! 
//! ### Local Update Batching:
//! 
//! Collecting updates into a local_updates vector for each thread reduces contention
//! on the mutex, as the updates are computed locally and applied in batches.
//! 
//! ## Drawbacks
//! 
//! ### Mutex Overhead:
//! 
//! Frequent locking and unlocking of the mutex can become a performance bottleneck, 
//! especially if threads contend for access to shared resources.
//! 
//! ### Reduced Scalability:
//! 
//! While parallel processing speeds up training, the mutex can limit scalability. 
//! As the number of threads increases, contention for the mutex can offset 
//! the benefits of parallelism.
//! 
//! ### Complexity:
//! 
//! Introducing shared state with Arc<Mutex<_>> adds complexity to the code, 
//! which could lead to subtle bugs or deadlocks if not managed carefully.
//! 
//! Text used from training  is from [SCOWL (Spell Checker Oriented Word Lists)](https://github.com/en-wl/wordlist)
//! and [dwyl/english-words](https://github.com/dwyl/english-words).
//! 
//! # Examples
//! 
//! ```
//! word2vec::example();
//! ```

use std::fs::File;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::io::{BufReader, BufWriter, BufRead, Read, Seek, SeekFrom};
use std::collections::{HashMap, HashSet};

use num_cpus;
use fastrand;
use serde_json;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};



/// Default paths for the model.
const DEFAULT_MODEL_PATH: &str = "models/word2vec_model.json";
/// Default path for the corpus.
const DEFAULT_CORPUS_PATH: &str = "data/word2vec_corpus.txt";

#[derive(Serialize, Deserialize)]
/// Structure for Word2Vec model.
/// 
/// This struct works with single thread. Suitable for small datasets.
/// 
/// # Fields
/// 
/// * `vocab` - Word to index mapping.
/// * `index_to_word` - Index to word mapping.
/// * `input_vectors` - Input layer embeddings.
/// * `output_vectors` - Output layer embeddings.
/// * `window_size` - Context window size.
/// * `embedding_dim` - Embedding dimension.
/// * `negative_samples` - Negative samples per positive sample.
/// 
/// # Examples
/// 
/// ```
/// let mut word2vec = SingleThreadWord2Vec::new(100, 50, 2, 5);
/// word2vec.train(&corpus, 10, 0.01);
/// word2vec.save(None);
/// ``` 
pub struct SingleThreadWord2Vec {
    vocab: HashMap<String, usize>,         // Word to index mapping
    index_to_word: Vec<String>,           // Index to word mapping
    input_vectors: Vec<Vec<f32>>,         // Input layer embeddings
    output_vectors: Vec<Vec<f32>>,        // Output layer embeddings
    window_size: usize,                   // Context window size
    embedding_dim: usize,                 // Embedding dimension
    negative_samples: usize,              // Negative samples per positive sample
}

impl SingleThreadWord2Vec {
    /// Initialize the Word2Vec model.
    /// 
    /// # Arguments
    /// 
    /// * `vocab_size` - Number of words in the vocabulary.
    /// * `embedding_dim` - Embedding dimension.
    /// * `window_size` - Context window size.
    /// * `negative_samples` - Negative samples per positive sample.
    ///
    /// # Returns
    /// 
    /// * `Self` - The `SingleThreadWord2Vec` instance.
    pub fn new(vocab_size: usize, embedding_dim: usize, window_size: usize, negative_samples: usize) -> Self {
        SingleThreadWord2Vec {
            vocab: HashMap::new(),
            index_to_word: Vec::new(),
            input_vectors: (0..vocab_size)
                .into_par_iter()
                .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                .collect(),
            output_vectors: (0..vocab_size)
                .into_par_iter()
                .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                .collect(),
            window_size,
            embedding_dim,
            negative_samples,
        }
    }

    /// Read the corpus from a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the corpus file.
    /// 
    /// # Returns
    /// 
    /// * `Result<Vec<String>, Box<dyn Error>>` - The corpus.
    pub fn read_corpus(path: Option<&str>) -> Result<Vec<String>, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_CORPUS_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().collect::<Result<Vec<_>, _>>()?)
    }

    /// Load the model from a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the model file.
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, Box<dyn Error>>` - The `SingleThreadWord2Vec` instance.
    pub fn load(path: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    /// Train the model using Skip-Gram with Negative Sampling.
    /// 
    /// # Arguments
    /// 
    /// * `corpus` - The corpus to train on.
    /// * `epochs` - The number of epochs to train for.
    /// * `learning_rate` - The learning rate.
    /// 
    /// # Examples
    /// 
    /// ```
    /// word2vec.train(&corpus, 10, 0.01);
    /// ```
    pub fn train(&mut self, corpus: &[String], epochs: usize, learning_rate: f32) {
        let pb = ProgressBar::new((epochs * corpus.len()) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap());

        for _epoch in 0..epochs {
            for (idx, word) in corpus.iter().enumerate() {
                if let Some(&word_idx) = self.vocab.get(word) {
                    let start = if idx >= self.window_size { idx - self.window_size } else { 0 };
                    let end = usize::min(idx + self.window_size + 1, corpus.len());

                    for context_word in &corpus[start..end] {
                        if context_word == word { continue; }
                        if let Some(&context_idx) = self.vocab.get(context_word) {
                            self.update_embeddings(word_idx, context_idx, 1.0, learning_rate);

                            let negative_samples: HashSet<usize> = (0..self.negative_samples)
                                .map(|_| fastrand::usize(0..self.vocab.len()))
                                .filter(|&idx| idx != context_idx)
                                .collect();

                            for &neg_idx in &negative_samples {
                                self.update_embeddings(word_idx, neg_idx, 0.0, learning_rate);
                            }
                        }
                    }
                }
                pb.inc(1);
            }
            pb.set_message(format!("Epoch {}/{}", _epoch + 1, epochs));
        }
        pb.finish_with_message("Training complete");
    }

    /// Update embeddings using gradient descent.
    /// 
    /// # Arguments
    /// 
    /// * `target_idx` - The index of the target word.
    /// * `context_idx` - The index of the context word.
    /// * `label` - The label for the update.
    /// * `learning_rate` - The learning rate.
    /// 
    /// # Examples
    /// 
    /// ```
    /// word2vec.update_embeddings(target_idx, context_idx, label, learning_rate);
    /// ```
    fn update_embeddings(&mut self, target_idx: usize, context_idx: usize, label: f32, learning_rate: f32) {
        let input_vec = self.input_vectors[target_idx].clone();
        let output_vec = self.output_vectors[context_idx].clone();

        let dot_product: f32 = input_vec.iter().zip(output_vec.iter()).map(|(x, y)| x * y).sum();
        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
        let error = sigmoid - label;

        // Sequential updates
        for i in 0..self.embedding_dim {
            self.input_vectors[target_idx][i] -= learning_rate * error * output_vec[i];
            self.output_vectors[context_idx][i] -= learning_rate * error * input_vec[i];
        }
    }

    /// Save the model to a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the model file.
    /// 
    /// # Returns
    /// 
    /// * `Result<(), Box<dyn Error>>` - The result of the operation.
    pub fn save(&self, path: Option<&str>) -> Result<(), Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &self)?;
        Ok(())
    }
}

/// Visitor for deserialization.
/// 
/// # Fields
/// 
/// * `vocab` - Word to index mapping.
/// * `index_to_word` - Index to word mapping.
/// 
/// # Examples
/// 
/// ```
/// let mut word2vec = SingleThreadWord2Vec::load(None).unwrap();
/// let mut multi_thread_word2vec = MultiThreadWord2Vec::load(None).unwrap();
/// ```
struct VisitorImpl<'a> {
    vocab: &'a mut Option<HashMap<String, usize>>,
    index_to_word: &'a mut Option<Vec<String>>,
}

impl<'de, 'a> serde::de::Visitor<'de> for VisitorImpl<'a> {
    type Value = ();

    /// Get the expected type.
    /// 
    /// # Arguments
    /// 
    /// * `formatter` - The formatter.
    /// 
    /// # Returns
    /// 
    /// * `std::fmt::Result` - The result of the operation.
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a Word2Vec model")
    }

    /// Visit the map.
    /// 
    /// # Arguments
    /// 
    /// * `map` - The map.
    /// 
    /// # Returns
    /// 
    /// * `Result<(), M::Error>` - The result of the operation.
    fn visit_map<M>(self, mut map: M) -> Result<(), M::Error>
    where
        M: serde::de::MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "vocab" => *self.vocab = Some(map.next_value()?),
                "index_to_word" => *self.index_to_word = Some(map.next_value()?),
                _ => { let _ = map.next_value::<serde::de::IgnoredAny>()?; }
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
/// Helper struct for serialization.
/// 
/// # Fields
/// 
/// * `vocab` - Word to index mapping.
/// * `index_to_word` - Index to word mapping.
/// * `input_vectors` - Input layer embeddings.
/// * `output_vectors` - Output layer embeddings.
/// * `window_size` - Context window size.
/// * `embedding_dim` - Embedding dimension.
/// * `negative_samples` - Negative samples per positive sample.
struct SerializableWord2Vec<'a> {
    vocab: &'a HashMap<String, usize>,
    index_to_word: &'a Vec<String>,
    input_vectors: &'a Vec<Vec<f32>>,
    output_vectors: &'a Vec<Vec<f32>>,
    window_size: usize,
    embedding_dim: usize,
    negative_samples: usize,
}

/// Structure for Word2Vec model.
/// 
/// This struct works with multi thread. Suitable for large datasets.
/// 
/// # Fields
/// 
/// * `vocab` - Word to index mapping.
/// * `index_to_word` - Index to word mapping.
/// * `input_vectors` - Input layer embeddings.
/// * `output_vectors` - Output layer embeddings.
/// * `window_size` - Context window size.
/// * `embedding_dim` - Embedding dimension.
/// * `negative_samples` - Negative samples per positive sample.
///
/// # Examples
/// 
/// ```
/// let mut word2vec = MultiThreadWord2Vec::new(100, 50, 2, 5);
/// word2vec.train(&corpus, 10, 0.01);
/// word2vec.save(None);
/// ```
pub struct MultiThreadWord2Vec {
    vocab: HashMap<String, usize>,          // Word to index mapping
    index_to_word: Vec<String>,            // Index to word mapping
    input_vectors: Arc<Mutex<Vec<Vec<f32>>>>, // Input layer embeddings
    output_vectors: Arc<Mutex<Vec<Vec<f32>>>>, // Output layer embeddings
    window_size: usize,                    // Context window size
    embedding_dim: usize,                  // Embedding dimension
    negative_samples: usize,               // Negative samples per positive sample
}

impl Serialize for MultiThreadWord2Vec {
    /// Serialize the model to a JSON file.
    /// 
    /// # Arguments
    /// 
    /// * `serializer` - The serializer.
    /// 
    /// # Returns
    /// 
    /// * `Result<S::Ok, S::Error>` - The result of the operation.  
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Word2Vec", 7)?;
        state.serialize_field("vocab", &self.vocab)?;
        state.serialize_field("index_to_word", &self.index_to_word)?;
        state.serialize_field("input_vectors", &*self.input_vectors.lock().unwrap())?;
        state.serialize_field("output_vectors", &*self.output_vectors.lock().unwrap())?;
        state.serialize_field("window_size", &self.window_size)?;
        state.serialize_field("embedding_dim", &self.embedding_dim)?;
        state.serialize_field("negative_samples", &self.negative_samples)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for MultiThreadWord2Vec {
    /// Deserialize the model from a JSON file.
    /// 
    /// # Arguments
    /// 
    /// * `deserializer` - The deserializer.
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, D::Error>` - The result of the operation.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::MapAccess;
        
        let mut vocab = None;
        let mut index_to_word = None;
        
        let mut map = deserializer.deserialize_struct("Word2Vec", 
            &["vocab", "index_to_word"], 
            VisitorImpl { 
                vocab: &mut vocab,
                index_to_word: &mut index_to_word,
            }
        )?;

        let vocab = vocab.ok_or_else(|| serde::de::Error::missing_field("vocab"))?;
        let index_to_word = index_to_word.ok_or_else(|| serde::de::Error::missing_field("index_to_word"))?;
        let vocab_size = vocab.len();
        let embedding_dim = 100; // Default value
        let window_size = 5; // Default value  
        let negative_samples = 5; // Default value

        Ok(MultiThreadWord2Vec {
            vocab,
            index_to_word,
            input_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            output_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            window_size,
            embedding_dim,
            negative_samples,
        })
    }
}

impl MultiThreadWord2Vec {
    /// Initialize the Word2Vec model.
    /// 
    /// # Arguments
    /// 
    /// * `vocab_size` - Number of words in the vocabulary.
    /// * `embedding_dim` - Embedding dimension.
    /// * `window_size` - Context window size.
    /// * `negative_samples` - Negative samples per positive sample.
    /// 
    /// # Returns
    /// 
    /// * `Self` - The `MultiThreadWord2Vec` instance.
    pub fn new(vocab_size: usize, embedding_dim: usize, window_size: usize, negative_samples: usize) -> Self {
        MultiThreadWord2Vec {
            vocab: HashMap::new(),
            index_to_word: Vec::new(),
            input_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            output_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            window_size,
            embedding_dim,
            negative_samples,
        }
        
    }

    /// Read the corpus from a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the corpus file.
    /// 
    /// # Returns
    /// 
    /// * `Result<Vec<String>, Box<dyn Error>>` - The corpus.
    pub fn read_corpus(path: Option<&str>) -> Result<Vec<String>, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_CORPUS_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().collect::<Result<Vec<_>, _>>()?)
    }

    /// Load the model from a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the model file.
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, Box<dyn Error>>` - The `MultiThreadWord2Vec` instance.
    pub fn load(path: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    /// Train the model using Skip-Gram with Negative Sampling.
    /// 
    /// # Arguments
    /// 
    /// * `corpus` - The corpus.
    /// * `epochs` - Number of epochs.
    /// * `learning_rate` - The learning rate.
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut word2vec = MultiThreadWord2Vec::new(100, 50, 2, 5);
    /// word2vec.train(&corpus, 10, 0.01);
    /// ```
    pub fn train(&mut self, corpus: &[String], epochs: usize, learning_rate: f32) {
        let pb = ProgressBar::new((epochs * corpus.len()) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap());

        for _epoch in 0..epochs {
            corpus.par_iter().enumerate().for_each(|(idx, word)| {
                if let Some(&word_idx) = self.vocab.get(word) {
                    let start = if idx >= self.window_size { idx - self.window_size } else { 0 };
                    let end = usize::min(idx + self.window_size + 1, corpus.len());
            
                    let mut local_updates = vec![];
                    for context_word in &corpus[start..end] {
                        if context_word == word { continue; }
                        if let Some(&context_idx) = self.vocab.get(context_word) {
                            local_updates.push((word_idx, context_idx, 1.0));
                            let negative_samples: HashSet<usize> = (0..self.negative_samples)
                                .map(|_| fastrand::usize(0..self.vocab.len()))
                                .filter(|&idx| idx != context_idx)
                                .collect();
            
                            for &neg_idx in &negative_samples {
                                local_updates.push((word_idx, neg_idx, 0.0));
                            }
                        }
                    }
            
                    // Apply updates sequentially
                    for (target_idx, context_idx, label) in local_updates {
                        self.update_embeddings(target_idx, context_idx, label, learning_rate);
                    }
                }
                pb.inc(1);
            });
            pb.set_message(format!("Epoch {}/{}", _epoch + 1, epochs));
        }
        pb.finish_with_message("Training complete");
    }

    /// Update embeddings using `gradient descent`.
    /// 
    /// # Arguments
    /// 
    /// * `target_idx` - The index of the target word.
    /// * `context_idx` - The index of the context word.
    /// * `label` - The label.
    /// * `learning_rate` - The learning rate.
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut word2vec = MultiThreadWord2Vec::new(100, 50, 2, 5);
    /// word2vec.update_embeddings(0, 1, 1.0, 0.01);
    /// ```
    fn update_embeddings(&self, target_idx: usize, context_idx: usize, label: f32, learning_rate: f32) {
        // Get input vectors - handle potential poison error
        let input_vec = match self.input_vectors.lock() {
            Ok(guard) => guard[target_idx].clone(),
            Err(poisoned) => poisoned.into_inner()[target_idx].clone(),
        };

        // Get output vectors - handle potential poison error
        let output_vec = match self.output_vectors.lock() {
            Ok(guard) => guard[context_idx].clone(),
            Err(poisoned) => poisoned.into_inner()[context_idx].clone(),
        };

        let dot_product: f32 = input_vec.iter().zip(output_vec.iter()).map(|(x, y)| x * y).sum();
        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
        let error = sigmoid - label;

        // Update vectors with error handling
        if let Ok(mut input_vectors) = self.input_vectors.lock() {
            for i in 0..self.embedding_dim {
                input_vectors[target_idx][i] -= learning_rate * error * output_vec[i];
            }
        }

        if let Ok(mut output_vectors) = self.output_vectors.lock() {
            for i in 0..self.embedding_dim {
                output_vectors[context_idx][i] -= learning_rate * error * input_vec[i];
            }
        }
    }
    
    /// Save the model to a file.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the model file.
    /// 
    /// # Returns
    /// 
    /// * `Result<(), Box<dyn Error>>` - The result of the operation.
    /// 
    /// # Examples
    /// 
    /// ```
    /// let mut word2vec = MultiThreadWord2Vec::new(100, 50, 2, 5);
    /// word2vec.save(Some("models/multi_thread_model.json")).unwrap();
    /// ```
    pub fn save(&self, path: Option<&str>) -> Result<(), Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Handle potential poison errors during serialization
        let input_vectors = self.input_vectors.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let output_vectors = self.output_vectors.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        
        let serializable_model = SerializableWord2Vec {
            vocab: &self.vocab,
            index_to_word: &self.index_to_word,
            input_vectors: &*input_vectors,
            output_vectors: &*output_vectors,
            window_size: self.window_size,
            embedding_dim: self.embedding_dim,
            negative_samples: self.negative_samples,
        };
        
        serde_json::to_writer(&mut writer, &serializable_model)?;
        Ok(())
    }
}


#[derive(Serialize, Deserialize)]
/// A struct to load a Word2Vec model from a file.
/// 
/// # Fields
/// 
/// * `vocab` - Word to index mapping.
/// * `index_to_word` - Index to word mapping.
/// * `input_vectors` - Input layer embeddings.
/// * `output_vectors` - Output layer embeddings.
/// * `window_size` - Context window size.
/// * `embedding_dim` - Embedding dimension.
/// * `negative_samples` - Negative samples per positive sample.
struct Word2VecFromFile{
    vocab: HashMap<String, usize>,        // Word to index mapping
    index_to_word: Vec<String>,           // Index to word mapping
    input_vectors: Vec<Vec<f32>>,         // Input layer embeddings
    output_vectors: Vec<Vec<f32>>,        // Output layer embeddings
    window_size: usize,                   // Context window size
    embedding_dim: usize,                 // Embedding dimension
    negative_samples: usize,              // Negative samples per positive sample
}

impl Word2VecFromFile {
    /// Read the JSON file in parallel.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the model file.
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, Box<dyn Error>>` - The `Word2VecFromFile` instance.
    pub fn read_json_parallel(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let chunk_size = file_size / num_cpus::get() as u64;
        
        let chunks: Vec<String> = (0..num_cpus::get() as u64)
            .into_par_iter()
            .map(|i| {
                let mut reader = BufReader::new(File::open(path).ok()?);
                let start = i * chunk_size;
                let end = if i == num_cpus::get() as u64 - 1 {
                    file_size
                } else {
                    (i + 1) * chunk_size
                };

                reader.seek(SeekFrom::Start(start)).ok()?;
                let mut buffer = vec![0; (end - start) as usize];
                reader.read_exact(&mut buffer).ok()?;

                String::from_utf8(buffer).ok()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("Failed to read file chunks")?;

        // Combine chunks and parse JSON
        let combined = chunks.join("");
        Ok(serde_json::from_str(&combined)?)
    }

    /// Create a new `Word2VecFromFile` instance.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the model file.
    /// 
    /// # Returns
    /// 
    /// * `Result<Self, Box<dyn Error>>` - The `Word2VecFromFile` instance.
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
        Self::read_json_parallel(path)
    }

    /// Get the embedding for a word.
    /// 
    /// # Arguments
    /// 
    /// * `word` - The word.
    /// 
    /// # Returns
    /// 
    /// * `Option<&[f32]>` - The embedding.
    pub fn get_embedding(&self, word: &str) -> Option<&[f32]> {
        let idx = self.vocab.get(word)?;
        Some(&self.input_vectors[*idx])
    }
}

/// Single-threaded example.
/// Using this means you have to train the model from scratch on a single thread.
pub fn single_thread_example() {
    // Example corpus.
    // As simple as it gets.
    println!("Reading corpus...");
    let corpus = vec![
        "the".to_string(),
        "quick".to_string(),
        "brown".to_string(),
        "fox".to_string(),
        "jumps".to_string(),
        "over".to_string(),
        "the".to_string(),
        "lazy".to_string(),
        "dog".to_string(),
    ];

    // Vocabulary setup.
    // The vocabulary size is the number of unique words in the corpus.
    // Any number of words can be used, but it must be greater than the number of unique words in the corpus.
    // Here we use 100, but it could be any number.
    let mut word2vec = SingleThreadWord2Vec::new(100, 50, 2, 5);
    let mut vocab_size = 0;
    for word in &corpus {
        word2vec.vocab.entry(word.clone()).or_insert_with(|| {
            let idx = vocab_size;
            vocab_size += 1;
            idx
        });
    }
    word2vec.index_to_word = word2vec.vocab.iter()
        .map(|(word, _)| word.clone())
        .collect();

    // Train the model.
    // The number of epochs and learning rate can be adjusted to improve the model's performance.
    println!("Training the model...");
    word2vec.train(&corpus, 10, 0.01); // 10 epochs, 0.01 learning rate

    // Print the embedding for a word.
    // We used plain vectors, so we can access the embedding directly.
    if let Some(&idx) = word2vec.vocab.get("fox") {
        println!("Embedding for 'fox': {:?}", word2vec.input_vectors[idx]);
    }

    // Save the model.
    // The model can be saved to a file using the `save` method.
    println!("Saving the model...");
    word2vec.save(Some("models/single_thread_model.json")).unwrap();
}


/// Multi-threaded example.
/// Using this means you can train the model on multiple threads.
pub fn multi_thread_example() {
    // Example corpus
    println!("Reading corpus...");
    let corpus = MultiThreadWord2Vec::read_corpus(None).unwrap();
    let vocab_size = corpus.len();

    // Vocabulary setup
    println!("Setting up vocabulary...");
    // vocab_size is the number of unique words in the corpus.
    // Any number of words can be used, but it must be greater than the number of unique words in the corpus.
    // Ideally,  the vocabulary size should be the number of words in the corpus.
    let mut word2vec = MultiThreadWord2Vec::new(vocab_size, 50, 2, 5);
    let mut vocab_size = 0;

    // Populate the vocabulary.
    for word in &corpus {
        word2vec.vocab.entry(word.clone()).or_insert_with(|| {
            let idx = vocab_size;
            vocab_size += 1;
            idx
        });
    }
    word2vec.index_to_word = word2vec.vocab.iter()
        .map(|(word, _)| word.clone())
        .collect();

    // Train the model.
    // The number of epochs and learning rate can be adjusted to improve the model's performance.
    println!("Training the model...");
    word2vec.train(&corpus, 10, 0.01); // 10 epochs, 0.01 learning rate

    // Print the embedding for a word.
    // since we used Arc to handle the vectors, we need to lock the access to the vectors.
    println!("Printing the embedding for 'fox'...");
    if let Some(&idx) = word2vec.vocab.get("fox") {
        println!("Embedding for 'fox': {:?}", word2vec.input_vectors.lock().unwrap()[idx]);
    }

    // Save the model.
    // The model can be saved to a file using the `save` method.
    println!("Saving the model...");
    word2vec.save(Some("models/multi_thread_model.json")).unwrap();
}


/// Load a model from a file.
/// You don't have to train the model from scratch, you can load a pre-trained model.
/// The model can be loaded from a file using the `new` method.
/// Once loaded, the model can be used to get the embedding for a word using the `get_embedding` method.
pub fn from_file_example(path: Option<  &str>) {
    println!("Loading model from file...");
    let model = Word2VecFromFile::new(path.unwrap_or("models/multi_thread_model.json")).unwrap();
    println!("Model loaded successfully.");

    println!("Getting embeddings for 'fox' and 'cow'...");
    println!("Embedding for 'fox': {:?}", model.get_embedding("fox"));
    println!("Embedding for 'cow': {:?}", model.get_embedding("cow"));
}