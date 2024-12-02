//! # Pattern Module
//! 
//! A module for analyzing patterns in text.
//! 
//! ## Features
//! 
//! - Generating n-grams from text
//! - Analyzing co-occurrences in text using a HybridNetwork
//! 
//! ##
//! Examples
//! 
//! ```
//! let n_grams = Pattern::n_grams("the quick brown fox jumps over the lazy dog", 2);
//! ```

#![allow(warnings)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::HashMap;
use crate::network::HybridNetwork;

/// A module for analyzing patterns in text.
/// 
/// # Examples
/// 
/// ```
/// use pattern::Pattern;
/// let n_grams = Pattern::n_grams("the quick brown fox jumps over the lazy dog", 2);
/// ```
pub struct Pattern {}
impl Pattern {

    /// Generates n-grams from the given text.
    /// 
    /// # Parameters
    /// 
    /// * text: &str
    /// 
    /// The input text to analyze.
    /// 
    /// * n: usize
    /// 
    /// The size of the n-grams to analyze.
    /// 
    /// # Returns
    /// 
    /// A HashMap containing the n-grams and their counts
    pub fn n_grams(text: &str, n: usize) -> HashMap<String, u32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams: HashMap<String, u32> = HashMap::new();
        
        // Generate n-grams
        for window in words.windows(n) {
            let ngram = window.join(" ");
            *ngrams.entry(ngram).or_insert(0) += 1;
        }     
        ngrams
    }
    
    /// Analyzes co-occurrences in the given text using a HybridNetwork.
    /// 
    /// # Parameters
    /// 
    /// * text: &str
    /// 
    /// The input text to analyze.
    /// 
    /// * n: usize
    /// 
    /// The size of the n-grams to analyze.
    /// 
    /// # Returns
    /// 
    /// A HybridNetwork containing the co-occurrence network
    pub fn co_occurrence_graph(text: &str, n: usize) -> HybridNetwork {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut network = HybridNetwork::new(words.len());

        // Add nodes for each unique word
        for (i, &word) in words.iter().enumerate() {
            network.add_node(i, word.to_string());
        }

        let mut co_occurrence_counts: HashMap<(usize, usize), u32> = HashMap::new();

        // Add edges for co-occurrences
        for window in words.windows(n) {

            if let [from, to] = window {
                let from_index = words.iter().position(|&w| w == *from).unwrap();
                let to_index = words.iter().position(|&w| w == *to).unwrap();
                *co_occurrence_counts.entry((from_index, to_index)).or_insert(0) += 1;
            }
        }

        for ((from_index, to_index), count) in co_occurrence_counts {
                network.add_edge(from_index, to_index, count as f64); // Weight can be adjusted
        }

        network
    }
}

pub fn example() {
    let text = "the quick brown fox jumps over the lazy dog";
    
    // Generate co-occurrence network
    let co_occurrence_network = Pattern::co_occurrence_graph(text, 2);
    
    // Example: Print the network's edges
    for (node, edges) in co_occurrence_network.edges.iter() {
        println!("Node {}: {:?}", node, edges);
    }
}