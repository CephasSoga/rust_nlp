use std::collections::HashMap;

pub struct Pattern {}
impl Pattern {
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
    
}



fn main() {
    let text = "the quick brown fox jumps over the lazy dog";
    
    // Generate different n-grams
    let bigrams = Pattern::n_grams(text, 2);
    let trigrams = Pattern::n_grams(text, 3);
    
    println!("Bigrams:");
    for (ngram, count) in &bigrams {
        println!("'{}': {}", ngram, count);
    }
    
    println!("\nTrigrams:");
    for (ngram, count) in &trigrams {
        println!("'{}': {}", ngram, count);
    }
}