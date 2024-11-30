//! # Hybrid Network
//! 
//! A hybrid network implementation that combines adjacency matrix and list representations.
//! 
//! This module provides a graph data structure that uses both an adjacency matrix for
//! fast edge lookups and an adjacency list for detailed edge information.
//! 
//! ## Examples
//! 
//! ```
//! use network::HybridNetwork;
//! 
//! let mut network = HybridNetwork::new(4);
//! network.add_node(0, "Start".to_string());
//! network.add_node(1, "End".to_string());
//! network.add_edge(0, 1, 1.0);
//! ```
//! 
//! ## Features
//! 
//! - O(1) edge weight lookups
//! - Detailed edge information storage
//! - Support for weighted directed graphs
//! - Common graph algorithms (BFS, Dijkstra's, etc.)
//! 
//! ## Comparison with Network
//! 
//! The `HybridNetwork` struct combines the advantages of both an adjacency matrix and an adjacency list:
//! 
//! - Fast edge weight lookups with O(1) complexity
//! - Detailed edge information storage for additional data
//! - Efficient space usage, similar to adjacency lists
//! - Flexibility in choosing between matrix and list representations
//! 
//! # Network
//! 
//! A simple adjacency list implementation for edge storage.
//! 
//! ## Examples
//! 
//! ```
//! use network::Network;
//! 
//! let mut network = Network::new();
//! network.add_node(0, "Start".to_string());
//! network.add_node(1, "End".to_string());
//! network.add_edge(0, 1, 1.0);
//! ```
//! 
//! ## Features
//! 
//! - Efficient space usage for sparse graphs
//! - Easy to implement common graph algorithms
//! - Suitable for operations that require detailed edge information
//!
//! ## Comparison with HybridNetwork
//! 
//! The `Network` struct provides a simpler and more space-efficient representation for sparse graphs:
//! 
//! - Faster for sparse graphs
//! - Simpler implementation
//! - Less memory overhead
//! 

#![allow(warnings)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
/// A node in the network
/// 
/// # Attributes
/// 
/// * id: usize
/// The unique identifier of the node
/// 
/// * data: String
/// Additional data associated with the node
pub struct Node {
    id: usize,
    data: String,
}

#[derive(Debug, Clone)]
/// An edge in the network
/// 
/// # Attributes
/// 
/// * from: usize
/// The starting node of the edge
/// 
/// * to: usize
/// The ending node of the edge
/// 
/// * weight: f64
/// The weight of the edge
pub struct Edge {
    from: usize,
    to: usize,
    weight: f64,
}


#[derive(Copy, Clone)]
/// A state used for Dijkstra's algorithm
/// 
/// # Attributes
/// 
/// * cost: f64
/// The cost to reach the node
/// 
/// * node: usize
/// The identifier of the node
struct State {
    cost: f64,
    node: usize,
}

impl PartialEq for State {
    /// Compares two State structs for equality
    /// 
    /// # Parameters
    /// 
    /// * other: &Self
    /// The other State struct to compare with
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for State {}

// Custom ordering for our priority queue
impl Ord for State {
    /// Compares two State structs for ordering
    /// 
    /// # Parameters
    /// 
    /// * other: &Self
    /// The other State struct to compare with
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.partial_cmp(&other.cost)
            .unwrap_or(Ordering::Equal)
            .reverse() // For min-heap
    }
}

impl PartialOrd for State {
    /// Compares two State structs for partial ordering
    /// 
    /// # Parameters
    /// 
    /// * other: &Self
    /// The other State struct to compare with  
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
/// A network that uses a fast lookup matrix for constant-time edge checks
/// and detailed edge information for additional data.
/// 
/// The term "hybrid" in this implementation refers to the dual data structure approach used to represent the graph/network:
/// 
/// 
/// * Adjacency Matrix (lookup: Vec<Vec<Option<f64>>>):
/// 
/// A 2D matrix where each cell [i][j] represents the weight of an edge from node i to node j
/// Provides O(1) edge weight lookups
/// Good for dense graphs
/// Uses more space (O(V²) where V is the number of vertices)
///
/// * Adjacency List (edges: HashMap<usize, Vec<Edge>>):
/// 
/// A map of nodes to their outgoing edges with detailed edge information
/// Better for iterating over a node's neighbors
/// More space-efficient for sparse graphs
/// Better for operations that need edge details
///
/// 
/// This hybrid approach combines the benefits of both representations:
/// Fast edge weight queries from the matrix
/// Detailed edge information and efficient iteration from the list
/// Trade-off between space and time complexity
/// Most graph implementations choose either an adjacency matrix OR an adjacency list, 
/// but this implementation uses both simultaneously, hence "hybrid". 
/// 
/// 
/// This can be particularly useful when you need both fast edge lookups and 
/// detailed edge information, though it does use more memory than either approach alone.
/// 
/// # Attributes
/// 
/// * lookup: Vec<Vec<Option<f64>>>
/// Fast lookup matrix for constant-time edge checks
/// 
/// * edges: HashMap<usize, Vec<Edge>>
/// Detailed edge information for additional data
/// 
/// * nodes: HashMap<usize, Node>
/// Node storage remains the same
/// 
/// * size: usize
/// Size of the network
/// 
/// # Methods
/// 
/// * new(size: usize) -> Self
/// Creates a new network with the given size
/// 
/// * ensure_capacity(id: usize)
/// Ensures the lookup matrix has space for the given node id
/// 
/// * add_node(id: usize, data: String)
/// Adds a node to the network
/// 
/// * add_edge(from: usize, to: usize, weight: f64)
/// Adds an edge to the network
///
/// * get_weight(from: usize, to: usize) -> Option<f64>
/// Returns the weight of the edge from node `from` to node `to`
/// 
/// * get_edge_details(from: usize) -> Option<&Vec<Edge>>
/// Returns detailed edge information for node `from`
///
/// * bfs(start: usize) -> Vec<usize>
/// Performs a breadth-first search starting from node `start`
///
/// * shortest_path(start: usize, end: usize) -> Option<(Vec<usize>, f64)>
/// Returns the shortest path and its distance between nodes `start` and `end`
///
/// * remove_node(id: usize) -> Option<Node>
/// Removes a node and all its associated edges
///
/// * remove_edge(from: usize, to: usize) -> Option<Edge>
/// Removes an edge between two nodes
///
/// * get_neighbors(id: usize) -> Vec<usize>
/// Returns all neighbors of a node
///
/// * degree(id: usize) -> usize
/// Returns the degree of a node (number of outgoing edges)
///
/// * has_path(start: usize, end: usize) -> bool
/// Checks if there exists a path between two nodes
/// 
/// * get_connected_component(start: usize) -> HashSet<usize>
/// Returns the strongly connected component containing the given node
/// using Kosaraju's algorithm (simplified version for demonstration)
/// 
/// # Example
/// 
/// ```
/// let mut network = HybridNetwork::new(4);
/// ```
pub struct HybridNetwork {
    // Fast lookup matrix for constant-time edge checks
    lookup: Vec<Vec<Option<f64>>>,
    // Detailed edge information for additional data
    edges: HashMap<usize, Vec<Edge>>,
    // Node storage remains the same
    nodes: HashMap<usize, Node>,
    size: usize,
}

impl HybridNetwork {
    /// Creates a new network with the given size
    /// 
    /// # Parameters
    /// 
    /// * size: usize
    /// The size of the network
    pub fn new(size: usize) -> Self {
        HybridNetwork {
            lookup: vec![vec![None; size]; size],
            edges: HashMap::new(),
            nodes: HashMap::new(),
            size,
        }
    }

    /// Ensures the lookup matrix has space for the given node id
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node
    pub fn ensure_capacity(&mut self, id: usize) {
        // If the id is greater than the current size, resize the lookup matrix
        if id >= self.size {
            let new_size = id + 1;
            // Resize existing rows
            for row in &mut self.lookup {
                row.resize(new_size, None);
            }
            // Add new rows
            while self.lookup.len() < new_size {
                self.lookup.push(vec![None; new_size]);
            }
            self.size = new_size;
        }
    }

    /// Adds a node to the network
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node
    /// 
    /// * data: String
    /// The data associated with the node   
    pub fn add_node(&mut self, id: usize, data: String) {
        self.ensure_capacity(id);
        self.nodes.insert(id, Node { id, data });
        self.edges.entry(id).or_insert(Vec::new());
    }

    /// Adds an edge to the network
    /// 
    /// # Parameters
    /// 
    /// * from: usize
    /// The id of the starting node
    /// 
    /// * to: usize
    /// The id of the ending node
    /// 
    /// * weight: f64
    /// The weight of the edge
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.ensure_capacity(from.max(to));
        // Update fast lookup matrix
        if from < self.size && to < self.size {
            self.lookup[from][to] = Some(weight);
        }

        // Store detailed edge information
        let edge = Edge { from, to, weight };
        self.edges.entry(from)
            .or_insert(Vec::new())
            .push(edge);
    }

    /// Fast edge weight lookup - O(1)
    /// 
    /// # Parameters
    /// 
    /// * from: usize
    /// The id of the starting node
    /// 
    /// * to: usize
    /// The id of the ending node
    pub fn get_weight(&self, from: usize, to: usize) -> Option<f64> {
        // Check if the ids are within the bounds of the lookup matrix
        if from < self.size && to < self.size {
            self.lookup[from][to]
        } else {
            None
        }
    }

    /// Get detailed edge information - O(deg(v))
    /// 
    /// # Parameters
    /// 
    /// * from: usize
    /// The id of the starting node
    pub fn get_edge_details(&self, from: usize) -> Option<&Vec<Edge>> {
        self.edges.get(&from)
    }

    /// Modified BFS using fast lookup
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            result.push(node);

            // Use fast lookup matrix for neighbor checking
            for to in 0..self.size {
                if self.lookup[node][to].is_some() && !visited.contains(&to) {
                    visited.insert(to);
                    queue.push_back(to);
                }
            }
        }

        result
    }

    /// Dijkstra's algorithm with a priority queue
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node
    /// 
    /// * end: usize
    /// The id of the ending node   
    pub fn shortest_path(&self, start: usize, end: usize) -> Option<(Vec<usize>, f64)> {
        // Initialize distance and previous maps
        let mut distances: HashMap<usize, f64> = HashMap::new();
        // Store the previous node for each node
        let mut previous: HashMap<usize, usize> = HashMap::new();
        // Initialize the priority queue
        let mut heap = BinaryHeap::new();

        // Initialize with start node
        distances.insert(start, 0.0);
        heap.push(State { cost: 0.0, node: start });

        // Process nodes in the priority queue
        while let Some(State { cost, node }) = heap.pop() {
            if node == end {
                break;
            }

            // Skip outdated entries
            if cost > *distances.get(&node).unwrap_or(&f64::INFINITY) {
                continue;
            }

            // Iterate over the neighbors of the current node
            if let Some(neighbors) = self.edges.get(&node) {
                for edge in neighbors {
                    let next = State {
                        cost: cost + edge.weight,
                        node: edge.to,
                    };

                    // Update the distance and previous node if a shorter path is found
                    if next.cost < *distances.get(&edge.to).unwrap_or(&f64::INFINITY) {
                        distances.insert(edge.to, next.cost);
                        previous.insert(edge.to, node);
                        heap.push(next);
                    }
                }
            }
        }

        // Reconstruct path
        if !previous.contains_key(&end) {
            return None;
        }
        // Reconstruct the path
        let mut path = vec![end];
        let mut current = end;
        while let Some(&prev) = previous.get(&current) {
            path.push(prev);
            current = prev;
        }
        path.reverse();

        Some((path, *distances.get(&end)?))
    }

    /// Removes a node and all its associated edges
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node
    pub fn remove_node(&mut self, id: usize) -> Option<Node> {
        if id >= self.size {
            return None;
        }

        // Remove from lookup matrix
        for row in &mut self.lookup {
            row[id] = None;
        }
        self.lookup[id].iter_mut().for_each(|x| *x = None);

        // Remove all edges containing this node
        self.edges.values_mut().for_each(|edges| {
            edges.retain(|edge| edge.to != id);
        });
        self.edges.remove(&id);

        // Remove and return the node
        self.nodes.remove(&id)
    }

    /// Removes an edge between two nodes
    /// 
    /// # Parameters
    /// 
    /// * from: usize
    /// The id of the starting node
    /// 
    /// * to: usize
    /// The id of the ending node
    pub fn remove_edge(&mut self, from: usize, to: usize) -> Option<Edge> {
        if from >= self.size || to >= self.size {
            return None;
        }

        // Remove from lookup matrix
        self.lookup[from][to] = None;

        // Remove from edges list
        if let Some(edges) = self.edges.get_mut(&from) {
            let position = edges.iter().position(|edge| edge.to == to)?;
            Some(edges.remove(position))
        } else {
            None
        }
    }

    /// Returns all neighbors of a node
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node  
    pub fn get_neighbors(&self, id: usize) -> Vec<usize> {
        // Check if the id is within the bounds of the lookup matrix
        if id >= self.size {
            return Vec::new();
        }

        // Get the neighbors of the node
        self.lookup[id]
            .iter()
            .enumerate()
            .filter_map(|(idx, &weight)| weight.map(|_| idx))
            .collect()
    }

    /// Returns the degree of a node (number of outgoing edges)
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node
    pub fn degree(&self, id: usize) -> usize {
        if id >= self.size {
            return 0;
        }
        self.lookup[id].iter().filter(|&&x| x.is_some()).count()
    }

    /// Checks if there exists a path between two nodes
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node
    /// 
    /// * end: usize
    /// The id of the ending node
    pub fn has_path(&self, start: usize, end: usize) -> bool {
        if start >= self.size || end >= self.size {
            return false;
        }

        // Initialize visited set and stack
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        // Process nodes in the stack
        while let Some(node) = stack.pop() {
            if node == end {
                return true;
            }

            // Add the node to the visited set
            if visited.insert(node) {
                // Add unvisited neighbors to stack
                for (next, &weight) in self.lookup[node].iter().enumerate() {
                    if weight.is_some() && !visited.contains(&next) {
                        stack.push(next);
                    }
                }
            }
        }

        false
    }

    /// Returns the strongly connected component containing the given node
    /// using Kosaraju's algorithm (simplified version for demonstration)
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node
    pub fn get_connected_component(&self, start: usize) -> HashSet<usize> {
        // Initialize the component set
        let mut component = HashSet::new();
        // Check if the id is within the bounds of the lookup matrix
        if start >= self.size {
            return component;
        }

        // Initialize visited set and stack
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        // Process nodes in the stack
        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                component.insert(node);
                // Add unvisited neighbors
                for (next, &weight) in self.lookup[node].iter().enumerate() {
                    if weight.is_some() && !visited.contains(&next) {
                        stack.push(next);
                    }
                }
            }
        }

        component
    }
}


#[derive(Debug, Clone)]
/// A network that uses a simple adjacency list for edge storage
/// 
/// # Attributes
/// 
/// * nodes: HashMap<usize, Node>
/// A map of node identifiers to their corresponding Node structs
/// 
/// * edges: HashMap<usize, Vec<Edge>>
/// A map of node identifiers to their outgoing edges
pub struct Network {
    nodes: HashMap<usize, Node>,
    edges: HashMap<usize, Vec<Edge>>,
}
impl Network { 
    /// Creates a new Network
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The new Network
    pub fn new() -> Self { 
        Network { 
            nodes: HashMap::new(), 
            edges: HashMap::new(), 
        } 
    } 
    
    /// Adds a node to the network
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node
    pub fn add_node(&mut self, id: usize, data: String) {
        let node = Node { id, data }; self.nodes.insert(id, node); 
    } 
    
    /// Adds an edge to the network
    /// 
    /// # Parameters
    /// 
    /// * from: usize
    /// The id of the starting node
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) { 
        let edge = Edge { from, to, weight }; 
        self.edges.entry(from).or_insert(Vec::new()).push(edge); 
    } 
    
    /// Breadth-First Search 
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node
    pub fn bfs(&self, start: usize) -> Vec<usize> { 
        let mut visited = HashSet::new(); 
        let mut queue = VecDeque::new(); 
        let mut result = Vec::new(); queue.push_back(start); 
        
        visited.insert(start); 
        
        while let Some(node) = queue.pop_front() { 
            result.push(node); 

            if let Some(neighbors) = self.edges.get(&node) { 
                for edge in neighbors { 
                    if !visited.contains(&edge.to) { 
                        visited.insert(edge.to); queue.push_back(edge.to); 
                    } 
                } 
            } 
        } 
        result 
    }
    
    /// Find shortest path using Dijkstra's algorithm 
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node
    pub fn shortest_path(&self, start: usize, end: usize) -> Option<(Vec<usize>, f64)> {
        let mut distances: HashMap<usize, f64> = HashMap::new(); 
        let mut previous: HashMap<usize, usize> = HashMap::new(); 
        let mut unvisited: HashSet<usize> = self.nodes.keys().cloned().collect(); 
        distances.insert(start, 0.0); 
        
        while !unvisited.is_empty() { 
            let current = unvisited .iter() .min_by(|&a, &b| { let da = distances.get(a).unwrap_or(&f64::INFINITY); 
                let db = distances.get(b).unwrap_or(&f64::INFINITY); da.partial_cmp(db).unwrap() }) .cloned()?; 
                
                if current == end { break; } unvisited.remove(&current); 
                if let Some(neighbors) = self.edges.get(&current) { 
                    
                    for edge in neighbors { 
                        if unvisited.contains(&edge.to) {
                            let alt = distances.get(&current).unwrap_or(&f64::INFINITY) + edge.weight; if alt < *distances.get(&edge.to).unwrap_or(&f64::INFINITY) {
                                distances.insert(edge.to, alt); previous.insert(edge.to, current); 
                            } 
                        } 
                    } 
                } 
            } 
            
            if !previous.contains_key(&end) { return None; } 
            
            let mut path = vec![end]; 
            let mut current = end; while let Some(&prev) = previous.get(&current) { path.push(prev); current = prev; } path.reverse(); Some((path, *distances.get(&end)?)) } fn shortest_path_efficient(&self, start: usize, end: usize) -> Option<(Vec<usize>, f64)> { let mut distances: HashMap<usize, f64> = HashMap::new(); let mut previous: HashMap<usize, usize> = HashMap::new(); 
            let mut heap = BinaryHeap::new(); 
                
            // Initialize with start node 
            distances.insert(start, 0.0);
            heap.push(State { cost: 0.0, node: start });
            while let Some(State { cost, node }) = heap.pop() {
                if node == end { break; } 
                // Skip outdated entries 
                if cost > *distances.get(&node).unwrap_or(&f64::INFINITY) { continue; } 
                if let Some(neighbors) = self.edges.get(&node) {
                     for edge in neighbors { 
                        let next = State { cost: cost + edge.weight, node: edge.to, }; 
                        if next.cost < *distances.get(&edge.to).unwrap_or(&f64::INFINITY) { 
                            distances.insert(edge.to, next.cost); 
                            previous.insert(edge.to, node); 
                            heap.push(next); 
                        } 
                    } 
                } 
            } 
            // Reconstruct path 
            if !previous.contains_key(&end) { return None; } 
            let mut path = vec![end]; 
            let mut current = end; 
            while let Some(&prev) = previous.get(&current) {
                path.push(prev); 
                current = prev; 
            } 
            path.reverse(); 
            Some((path, *distances.get(&end)?)) 
        } 
    } 
    


fn main() {

    // ###Hybrid Network###
    let mut network = HybridNetwork::new(4);
    
    // Add nodes
    network.add_node(1, "Node 1".to_string());
    network.add_node(2, "Node 2".to_string());
    network.add_node(3, "Node 3".to_string());
    network.add_node(4, "Node 4".to_string());

    // Add edges
    network.add_edge(1, 2, 4.0);
    network.add_edge(1, 3, 2.0);
    network.add_edge(2, 4, 3.0);
    network.add_edge(3, 4, 1.0);

    // Test BFS
    println!("BFS from node 1: {:?}", network.bfs(1));

    // Test shortest path
    if let Some((path, distance)) = network.shortest_path(1, 4) {
        println!("Shortest path from 1 to 4: {:?}", path);
        println!("Total distance: {}", distance);
    }

    // ###Network###
    let mut network = Network::new(); 
    // Add nodes 
    network.add_node(1, "Node 1".to_string()); 
    network.add_node(2, "Node 2".to_string()); 
    network.add_node(3, "Node 3".to_string()); 
    network.add_node(4, "Node 4".to_string()); 
    // Add edges 
    network.add_edge(1, 2, 4.0); 
    network.add_edge(1, 3, 2.0); 
    network.add_edge(2, 4, 3.0); 
    network.add_edge(3, 4, 1.0); 

    // Test BFS 
    println!("BFS from node 1: {:?}", network.bfs(1)); 
    // Test shortest path 
    if let Some((path, distance)) = network.shortest_path(1, 4) { 
        println!("Shortest path from 1 to 4: {:?}", path); 
        println!("Total distance: {}", distance); 
    }
}