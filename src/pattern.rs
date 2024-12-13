//! # Pattern Module.
//! 
//! A module for analyzing patterns in text.
//! 
//! ## Key Features
//! 
//! - N-Gram Generation
//! 
//! The n_grams method efficiently creates n-grams and counts their occurrences in 
//! the given text.
//! Example usage in documentation is clear.
//! 
//! - Co-Occurrence Graph
//! 
//! co_occurrence_graph builds a HybridNetwork by analyzing word 
//! co-occurrences in sliding windows of size n.
//! Integrates seamlessly with the HybridNetwork class.
//! 
//! - Temporal Analysis
//! 
//! n_grams_over_time and n_gram_frequency_over_time group and analyze n-grams 
//! across multiple time periods.
//! These methods provide both counts and normalized frequencies, 
//! making them suitable for trend analysis.
//! 
//! ## Statistics on N-grams
//! 
//! - Mean (μ)
//!   * Formula: μ = (Σx) / n
//!   * Where x are the frequency values and n is the number of values
//! 
//! - Median
//!   * Middle value when frequencies are sorted
//!   * For even n: average of two middle values
//! 
//! - Range
//!   * Formula: max(x) - min(x)
//!   * Difference between highest and lowest frequencies
//! 
//! - Growth Rate
//!   * Formula: ((x₂ - x₁) / x₁) * 100%
//!   * Where x₁ is initial frequency and x₂ is final frequency
//! 
//! - Correlation (Pearson)
//!   * Formula: r = Σ((x - μₓ)(y - μᵧ)) / √(Σ(x - μₓ)² * Σ(y - μᵧ)²)
//!   * Measures linear relationship between two n-gram frequencies
//! 
//! - Probability Distribution
//!   * Represents frequency of each n-gram
//!   * Useful for understanding distribution patterns
//!   * Formula: P(x) = count(x) / total_count
//! 
//! - Entropy
//!   * Measures unpredictability of n-gram distribution
//!   * Formula: H(X) = -Σ(P(x) * log₂(P(x)))
//! 
//! - KL Divergence
//!   * Compares two probability distributions
//!   * Formula: D(P || Q) = Σ(P(x) * log₂(P(x) / Q(x)))
//! 
//! - JS Divergence
//!   * Similarity measure between two distributions
//!   * Formula: JS(P || Q) = 0.5 * (KL(P || M) + KL(Q || M))
//!   * Where M is the average of P and Q
//! 
#![allow(warnings)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::HashMap;

use rayon::prelude::*;

use crate::network::HybridNetwork;


/// A module for analyzing patterns in text.
pub struct Pattern {}
impl Pattern {

    /// Generate n-grams from a given text.
    /// 
    /// ## Arguments
    /// 
    /// * `text` - The text to generate n-grams from.
    /// * `n` - The size of the n-grams to generate.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing n-grams as keys and their counts as values.
    pub fn n_grams(text: &str, n: usize) -> HashMap<String, u32> {
        // Split the text into words
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams: HashMap<String, u32> = HashMap::new();
        
        // Generate n-grams
        for window in words.windows(n) {
            let ngram = window.join(" ");
            *ngrams.entry(ngram).or_insert(0) += 1;
        }     
        ngrams
    }


    /// Generate a co-occurrence graph from a given text.
    /// 
    /// ## Arguments
    /// 
    /// * `text` - The text to generate a co-occurrence graph from.
    /// * `n` - The size of the n-grams to generate.
    /// 
    /// ## Returns
    /// 
    /// A `HybridNetwork` containing the co-occurrence graph.
    pub fn co_occurrence_graph(text: &str, n: usize) -> HybridNetwork {
        // Split the text into words
        let words: Vec<&str> = text.split_whitespace().collect();
        // Create a new HybridNetwork with the number of unique words
        let mut network = HybridNetwork::new(words.len());
    
        // Precompute word indices for better performance
        let word_indices: HashMap<&str, usize> = words.iter()
            .enumerate()
            .map(|(i, &word)| (word, i))
            .collect();
    
        // Initialize a HashMap to store co-occurrence counts
        let mut co_occurrence_counts: HashMap<(usize, usize), u32> = HashMap::new();
    
        // Iterate over sliding windows of size n
        for window in words.windows(n) {
            // Iterate over all pairs of words in the current window
            for i in 0..window.len() {
                for j in (i + 1)..window.len() {
                    // Get the indices of the current pair of words
                    let from = word_indices[window[i]];
                    let to = word_indices[window[j]];
                    // Increment the co-occurrence count for the pair
                    *co_occurrence_counts.entry((from, to)).or_insert(0) += 1;
                }
            }
        }
    
        for ((from_index, to_index), count) in co_occurrence_counts {
            network.add_edge(from_index, to_index, count as f64);
        }
    
        network
    }
    

    /// Generate n-grams over time.
    /// 
    /// ## Arguments
    /// 
    /// * `texts` - A vector of tuples containing the text and time period.
    /// * `n` - The size of the n-grams to generate.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing the time periods as keys and n-grams as values.
    pub fn n_grams_count_over_time(texts: Vec<(&str, &str)>, n: usize) -> HashMap<String, HashMap<String, u32>> {
        let mut time_period_ngrams: HashMap<String, HashMap<String, u32>> = HashMap::new();

        for (text, time_period) in texts {
            let ngrams = Pattern::n_grams(text, n);
            time_period_ngrams.insert(time_period.to_string(), ngrams);
        }

        time_period_ngrams
    }

    /// Calculate the frequency of n-grams over time.
    /// 
    /// ## Arguments
    /// 
    /// * `texts` - A vector of tuples containing the text and time period.
    /// * `n` - The size of the n-grams to generate.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing the time periods as keys and n-gram frequencies as values.
    pub fn n_gram_frequency_over_time(texts: Vec<(&str, &str)>, n: usize) -> HashMap<String, HashMap<String, f64>> {
        // First get n-gram counts over time
        let ngrams = Pattern::n_grams_count_over_time(texts, n);
        
        // Calculate frequencies for each time period
        let mut frequencies: HashMap<String, HashMap<String, f64>> = HashMap::new();
        
        // Iterate over each time period and calculate frequencies
        for (period, counts) in ngrams {
            let total: u32 = counts.values().sum();
            let total_f64 = total as f64;
            
            // Calculate frequencies for each n-gram in the current time period
            let period_frequencies: HashMap<String, f64> = counts.into_iter()
                .map(|(ngram, count)| {
                    (ngram, count as f64 / total_f64)
                })
                .collect();
                
            frequencies.insert(period, period_frequencies);
        }
        
        frequencies
    }

    /// Generate n-grams over time in parallel.
    /// 
    /// ## Arguments
    /// 
    /// * `texts` - A vector of tuples containing the text and time period.
    /// * `n` - The size of the n-grams to generate.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing the time periods as keys and n-grams as values.
    pub fn n_grams_count_over_time_parallel(texts: Vec<(&str, &str)>, n: usize) -> HashMap<String, HashMap<String, u32>> {
        // Generate n-grams in parallel by mapping over the texts
        texts.par_iter()
            .map(|(text, time_period)| {
                let ngrams = Pattern::n_grams(text, n);
                (time_period.to_string(), ngrams)
            })
            .collect()
    }

    /// Calculate the frequency of n-grams over time in parallel.
    /// 
    /// ## Arguments
    /// 
    /// * `texts` - A vector of tuples containing the text and time period.
    /// * `n` - The size of the n-grams to generate.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing the time periods as keys and n-gram frequencies as values.
    pub fn n_gram_frequency_over_time_parallel(texts: Vec<(&str, &str)>, n: usize) -> HashMap<String, HashMap<String, f64>> {
        // First get n-gram counts in parallel
        let ngrams: HashMap<String, HashMap<String, u32>> = texts.par_iter()
            .map(|(text, time_period)| {
                let ngrams = Pattern::n_grams(text, n);
                (time_period.to_string(), ngrams)
            })
            .collect();

        // Calculate frequencies in parallel
        let frequencies: HashMap<String, HashMap<String, f64>> = ngrams.par_iter()
            .map(|(period, counts)| {
                let total: u32 = counts.values().sum();
                let total_f64 = total as f64;
                
                // Calculate frequencies for each n-gram in the current time period
                let period_frequencies: HashMap<String, f64> = counts.iter()
                    .map(|(ngram, count)| {
                        (ngram.clone(), *count as f64 / total_f64)
                    })
                    .collect();

                // Return the time period and its frequencies
                (period.clone(), period_frequencies)
            })
            .collect();

        frequencies
    }
}


/// A module for analyzing statistics on n-grams.
pub struct PatternStats {}

impl PatternStats {

    /// Calculate the mean of a given data set.
    /// 
    /// ## Formula   
    /// 
    /// μ = (Σx) / n
    /// 
    /// Where x are the frequency values and n is the number of values.
    /// 
    /// ## Arguments
    /// 
    /// * `data` - A vector of u32 values.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the mean.
    pub fn mean(data: &[u32]) -> f64 {
        let sum: u32 = data.iter().sum();
        sum as f64 / data.len() as f64
    }

    /// Calculate the variance of a given data set.
    /// 
    /// ## Formula
    /// 
    /// σ² = Σ((x - μ)²) / n
    /// 
    /// Where x are the frequency values, μ is the mean, and n is the number of values.
    /// 
    /// ## Arguments
    /// 
    /// * `data` - A vector of u32 values.
    /// * `mean` - A `f64` value representing the mean.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the variance.
    pub fn variance(data: &[u32], mean: f64) -> f64 {
        data.iter()
            .map(|&count| (count as f64 - mean).powi(2))
            .sum::<f64>() / data.len() as f64
    }

    /// Calculate the standard deviation of a given data set.
    /// 
    /// ## Formula
    /// 
    /// σ = √(σ²)
    /// 
    /// Where σ² is the variance.
    /// 
    /// ## Arguments
    /// 
    /// * `variance` - A `f64` value representing the variance.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the standard deviation.
    pub fn std_deviation(variance: f64) -> f64 {
        variance.sqrt()
    }

    /// Calculate the covariance of two given data sets.
    /// 
    /// ## Formula
    /// 
    /// cov(x, y) = Σ((x - μₓ)(y - μᵧ)) / n
    /// 
    /// Where x and y are the frequency values, μₓ is the mean of x, and μᵧ is the mean of y.
    /// 
    /// ## Arguments
    /// 
    /// * `data_x` - A vector of u32 values.
    /// * `data_y` - A vector of u32 values.
    /// * `mean_x` - A `f64` value representing the mean of `data_x`.
    /// * `mean_y` - A `f64` value representing the mean of `data_y`.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the covariance.
    pub fn covariance(data_x: &[u32], data_y: &[u32], mean_x: f64, mean_y: f64) -> f64 {
        data_x.iter()
            .zip(data_y.iter())
            .map(|(&x, &y)| (x as f64 - mean_x) * (y as f64 - mean_y))
            .sum::<f64>() / data_x.len() as f64
    }

    /// Calculate the median of a given data set.
    /// 
    /// ## Formula
    /// 
    /// If n is even, median = (data[n/2 - 1] + data[n/2]) / 2
    /// If n is odd, median = data[n/2]
    /// 
    /// Where n is the number of values in the data set.
    /// 
    /// ## Arguments
    /// 
    /// * `data` - A vector of u32 values.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the median.
    pub fn median(mut data: Vec<u32>) -> f64 {
        data.sort();
        let len = data.len();
        if len % 2 == 0 {
            (data[len / 2 - 1] as f64 + data[len / 2] as f64) / 2.0
        } else {
            data[len / 2] as f64
        }
    }

    /// Calculate the range of a given data set.
    /// 
    /// ## Formula
    /// 
    /// range = max(data) - min(data)
    /// 
    /// Where max(data) is the maximum value in the data set and min(data) is the minimum value in the data set.
    /// 
    /// ## Arguments
    /// 
    /// * `data` - A vector of u32 values.
    /// 
    /// ## Returns
    /// 
    /// A `u32` value representing the range.
    pub fn range(data: &[u32]) -> u32 {
        data.iter().max().unwrap() - data.iter().min().unwrap()
    }

    /// Calculate the growth rate of a given data set.
    /// 
    /// ## Formula
    /// 
    /// growth rate = ((x₂ - x₁) / x₁) * 100%
    /// 
    /// Where x₁ is the initial frequency and x₂ is the final frequency.
    /// 
    /// ## Arguments
    /// 
    /// * `data` - A vector of u32 values.
    /// 
    /// ## Returns
    /// 
    /// A vector of `f64` values representing the growth rate.  
    pub fn growth_rate(data: &[u32]) -> Vec<f64> {
        data.windows(2)
            .map(|window| ((window[1] as f64 - window[0] as f64) / window[0] as f64) * 100.0)
            .collect()
    }

    /// Calculate the correlation between two given data sets.
    /// 
    /// ## Formula
    /// 
    /// corr(x, y) = covariance(x, y) / (std_dev(x) * std_dev(y))
    /// 
    /// Where covariance(x, y) is the covariance between x and y, std_dev(x) is the standard deviation of x, and std_dev(y) is the standard deviation of y.
    /// 
    /// ## Arguments
    /// 
    /// * `data_x` - A vector of u32 values.
    /// * `data_y` - A vector of u32 values.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the correlation.
    pub fn correlation(data_x: &[u32], data_y: &[u32]) -> f64 {
        let mean_x = Self::mean(data_x);
        let mean_y = Self::mean(data_y);
        let std_x = Self::std_deviation(Self::variance(data_x, mean_x));
        let std_y = Self::std_deviation(Self::variance(data_y, mean_y));
        let covariance = Self::covariance(data_x, data_y, mean_x, mean_y);

        covariance / (std_x * std_y)
    }

    /// Calculate the probability distribution of a given data set.
    /// 
    /// ## Formula
    /// 
    /// p(x) = count(x) / Σ(count(x))
    /// 
    /// Where count(x) is the count of x and Σ(count(x)) is the sum of all counts.
    /// 
    /// ## Arguments
    /// 
    /// * `ngrams` - A `HashMap` containing n-grams as keys and their counts as values.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing n-grams as keys and their probabilities as values.   
    pub fn probability_distribution(ngrams: &HashMap<String, u32>) -> HashMap<String, f64> {
        let total_count: u32 = ngrams.values().sum();
        ngrams.iter()
            .map(|(ngram, &count)| (ngram.clone(), count as f64 / total_count as f64))
            .collect()
    }

    /// Calculate the entropy of a given probability distribution.
    /// 
    /// ## Formula
    /// 
    /// H(X) = -Σ(p(x) * log₂(p(x)))
    /// 
    /// Where p(x) is the probability of x.
    /// 
    /// ## Arguments
    /// 
    /// * `prob_dist` - A `HashMap` containing n-grams as keys and their probabilities as values.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the entropy.
    pub fn entropy(prob_dist: &HashMap<String, f64>) -> f64 {
        prob_dist.values()
            .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum()
    }

    /// Calculate the KL divergence between two given probability distributions.
    /// 
    /// ## Formula
    /// 
    /// D(P || Q) = Σ(p(x) * log₂(p(x) / q(x)))
    /// 
    /// Where p(x) is the probability of x in the first distribution and q(x) is the probability of x in the second distribution.
    /// 
    /// ## Arguments
    /// 
    /// * `p` - A `HashMap` containing n-grams as keys and their probabilities as values.
    /// * `q` - A `HashMap` containing n-grams as keys and their probabilities as values.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the KL divergence.
    pub fn kl_divergence(p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
        p.iter()
            .map(|(ngram, &p_prob)| {
                if let Some(&q_prob) = q.get(ngram) {
                    if p_prob > 0.0 && q_prob > 0.0 {
                        p_prob * (p_prob / q_prob).log2()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Calculate the Jensen-Shannon divergence between two given probability distributions.
    /// 
    /// ## Formula
    /// 
    /// D(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    /// 
    /// Where M is the average of P and Q.
    /// 
    /// ## Arguments
    /// 
    /// * `p` - A `HashMap` containing n-grams as keys and their probabilities as values.
    /// * `q` - A `HashMap` containing n-grams as keys and their probabilities as values.
    /// 
    /// ## Returns
    /// 
    /// A `f64` value representing the Jensen-Shannon divergence.
    pub fn js_divergence(p: &HashMap<String, f64>, q: &HashMap<String, f64>) -> f64 {
        let m: HashMap<String, f64> = p.iter()
            .map(|(ngram, &p_prob)| {
                let q_prob = *q.get(ngram).unwrap_or(&0.0);
                (ngram.clone(), (p_prob + q_prob) / 2.0)
            })
            .collect();

        0.5 * Self::kl_divergence(p, &m) + 0.5 * Self::kl_divergence(q, &m)
    }

    /// Find the mode of a given probability distribution.
    /// 
    /// ## Formula
    /// 
    /// mode = argmax(p(x))
    /// 
    /// Where p(x) is the probability of x.
    /// 
    /// ## Arguments
    /// 
    /// * `prob_dist` - A `HashMap` containing n-grams as keys and their probabilities as values.
    /// 
    /// ## Returns
    /// 
    /// An `Option<String>` representing the mode.
    pub fn mode(prob_dist: &HashMap<String, f64>) -> Option<String> {
        prob_dist.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(ngram, _)| ngram.clone())
    }

    /// Analyze n-grams over time.
    /// 
    /// ## Arguments
    /// 
    /// * `ngrams_over_time` - A `HashMap` containing time periods as keys and n-grams as values.
    /// 
    /// ## Returns
    /// 
    /// A `HashMap` containing n-grams as keys and their statistics as values.
    pub fn analyze_ngrams(ngrams_over_time: HashMap<String, HashMap<String, u32>>) -> HashMap<String, HashMap<String, f64>> {
        let mut stats: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for (ngram, time_counts) in ngrams_over_time.iter() {
            let counts: Vec<u32> = time_counts.values().cloned().collect();
            let mean = Self::mean(&counts);
            let variance = Self::variance(&counts, mean);
            let std_dev = Self::std_deviation(variance);

            stats.insert(
                ngram.clone(),
                HashMap::from([
                    ("mean".to_string(), mean),
                    ("variance".to_string(), variance),
                    ("std_dev".to_string(), std_dev),
                ]),
            );
        }

        stats
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

    let ngrams_over_time = HashMap::new();

    let stats = PatternStats::analyze_ngrams(ngrams_over_time);
}
