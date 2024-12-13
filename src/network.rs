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

use std::error::Error;

use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;

use plotters::prelude::*;
use eframe::egui;
use egui_plot::{Plot, Points, Line, PlotPoints};

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
    /// Compares two State structs for equality.
    /// 
    /// # Parameters
    /// 
    /// * other: &Self
    /// The other State struct to compare with.
    /// 
    /// # Returns
    /// 
    /// * bool
    /// True if the two State structs are equal, false otherwise.
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for State {}

// Custom ordering for our priority queue
impl Ord for State {
    /// Compares two State structs for ordering.
    /// 
    /// # Parameters
    /// 
    /// * other: &Self
    /// The other State struct to compare with.
    /// 
    /// # Returns
    /// 
    /// * Ordering
    /// The ordering of the two State structs.
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.partial_cmp(&other.cost)
            .unwrap_or(Ordering::Equal)
            .reverse() // For min-heap
    }
}

impl PartialOrd for State {
    /// Compares two State structs for partial ordering.
    /// 
    /// # Parameters
    /// 
    /// * other: &Self
    /// The other State struct to compare with.
    /// 
    /// # Returns
    /// 
    /// * Option<Ordering>
    /// The partial ordering of the two State structs.
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
    pub lookup: Vec<Vec<Option<f64>>>,
    // Detailed edge information for additional data
    pub edges: HashMap<usize, Vec<Edge>>,
    // Node storage remains the same
    pub nodes: HashMap<usize, Node>,
    pub size: usize,
}

impl HybridNetwork {
    /// Creates a new network with the given size.
    /// 
    /// # Parameters
    /// 
    /// * size: usize
    /// The size of the network.
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The new network.
    pub fn new(size: usize) -> Self {
        HybridNetwork {
            lookup: vec![vec![None; size]; size],
            edges: HashMap::new(),
            nodes: HashMap::new(),
            size,
        }
    }

    /// Ensures the lookup matrix has space for the given node id.
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node.
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The updated network.
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
    /// The id of the node.
    /// 
    /// * data: String
    /// The data associated with the node.
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The updated network.
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
        /// The id of the starting node.
    /// 
    /// * to: usize
    /// The id of the ending node.
    /// 
    /// * weight: f64
    /// The weight of the edge.
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The updated network.
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
    /// The id of the starting node.
    /// 
    /// * to: usize
    /// The id of the ending node.
    /// 
    /// # Returns
    /// 
    /// * Option<f64>
    /// The weight of the edge from node `from` to node `to`.
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
        /// The id of the starting node.
    /// 
    /// # Returns
    /// 
    /// * Option<&Vec<Edge>>
    /// The detailed edge information for the node.
    pub fn get_edge_details(&self, from: usize) -> Option<&Vec<Edge>> {
        self.edges.get(&from)
    }

    /// Modified BFS using fast lookup
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node.
    /// 
    /// # Returns
    /// 
    /// * Vec<usize>
    /// The BFS traversal order.
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

    /// Dijkstra's algorithm with a priority queue.
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node.    
    /// 
    /// * end: usize
    /// The id of the ending node.
    /// 
    /// # Returns
    /// 
    /// * Option<(Vec<usize>, f64)>
    /// The shortest path and its distance between nodes `start` and `end`.
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

    /// Removes a node and all its associated edges.
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node.
    /// 
    /// # Returns
    /// 
    /// * Option<Node>
    /// The removed node.
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

    /// Removes an edge between two nodes.
    /// 
    /// # Parameters
    /// 
    /// * from: usize
    /// The id of the starting node
    /// 
    /// * to: usize
    /// The id of the ending node.
    /// 
    /// # Returns
    /// 
    /// * Option<Edge>
    /// The removed edge.
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

    /// Returns all neighbors of a node.
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node.
    /// 
    /// # Returns
    /// 
    /// * Vec<usize>
    /// The neighbors of the node.
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

    /// Returns the degree of a node (number of outgoing edges).
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node.
    /// 
    /// # Returns
    /// 
    /// * usize
    /// The degree of the node.
    /// 
    /// # Notes
    /// 
    /// * This method is O(n) where n is the number of nodes in the network.
    pub fn degree(&self, id: usize) -> usize {
        if id >= self.size {
            return 0;
        }
        self.lookup[id].iter().filter(|&&x| x.is_some()).count()
    }

    /// Checks if there exists a path between two nodes.
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node.
    /// 
    /// * end: usize
    /// The id of the ending node.
    /// 
    /// # Returns
    /// 
    /// * bool
    /// True if there exists a path between the two nodes, false otherwise.
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
    /// using Kosaraju's algorithm (simplified version for demonstration).
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node.
    /// 
    /// # Returns
    /// 
    /// * HashSet<usize>
    /// The strongly connected component containing the given node. 
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

    pub fn static_plot(&self, output_file: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(0f64..800f64, 0f64..600f64)?;

        let nodes: Vec<(f64, f64)> = (0..self.size)
            .map(|_| (
                fastrand::f64() * 700.0 + 50.0,
                fastrand::f64() * 500.0 + 50.0
            ))
            .collect();

        // Draw edges with chart context
        for i in 0..self.size {
            for j in 0..self.size {
                if let Some(weight) = self.lookup[i][j] {
                    let color = if weight > 0.5 { RED } else { BLUE };
                    chart.plotting_area().draw(&PathElement::new(
                        vec![(nodes[i].0, nodes[i].1), (nodes[j].0, nodes[j].1)],
                        color.mix(0.5),
                    ))?;
                }
            }
        }

        // Draw nodes with chart context
        for (i, (x, y)) in nodes.iter().enumerate() {
            chart.plotting_area().draw(&Circle::new(
                (*x, *y),
                5,
                ShapeStyle::from(&BLACK).filled(),
            ))?;
            
            if let Some(node) = self.nodes.get(&i) {
                chart.plotting_area().draw(&Text::new(
                    node.data.clone(),
                    (*x + 10.0, *y),
                    ("sans-serif", 15).into_font(),
                ))?;
            }
        }

        Ok(())
    }

    pub fn show_interactive(&self) -> Result<(), eframe::Error> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([800.0, 600.0]),
            ..Default::default()
        };

        eframe::run_native(
            "Network Visualization",
            options,
            Box::new(|_cc| Ok(Box::new(NetworkVisualizer::new(self))))
        )
    }
}


/// A visualizer for the HybridNetwork struct.
/// 
/// # Attributes
/// 
/// * network: &'a HybridNetwork
/// The network to visualize.
/// 
/// * node_positions: HashMap<usize, [f64; 2]>
/// The positions of the nodes.
/// 
/// * selected_node: Option<usize>
/// The id of the selected node.
/// 
/// * dragging: Option<usize>
/// The id of the node being dragged.
struct NetworkVisualizer<'a> {
    network: &'a HybridNetwork,
    node_positions: HashMap<usize, [f64; 2]>,
    selected_node: Option<usize>,
    dragging: Option<usize>,
}

impl<'a> NetworkVisualizer<'a> {
    fn new(network: &'a HybridNetwork) -> Self {
        let mut node_positions = HashMap::new();
        for i in 0..network.size {
            node_positions.insert(i, [
                fastrand::f64() * 700.0 + 50.0,
                fastrand::f64() * 500.0 + 50.0
            ]);
        }
        Self {
            network,
            node_positions,
            selected_node: None,
            dragging: None,
        }
    }
}

impl<'a> eframe::App for NetworkVisualizer<'a> {
    /// Updates the network visualizer.
    /// 
    /// # Parameters
    /// 
    /// * ctx: &egui::Context
    /// The context of the application.
    /// 
    /// * _frame: &mut eframe::Frame
    /// The frame of the application.
    /// 
    /// # Returns
    /// 
    /// * ()
    /// The updated network visualizer. 
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let input = ui.input(|i| i.clone());
            
            let plot = Plot::new("network_plot")
                .allow_drag(false)
                .allow_zoom(true)
                .include_y(0.0)
                .include_y(600.0)
                .include_x(0.0)
                .include_x(800.0);

            // Track if we clicked near any node
            let mut clicked_node = None;

            plot.show(ui, |plot_ui| {
                // Draw edges first
                for i in 0..self.network.size {
                    for j in 0..self.network.size {
                        if let Some(weight) = self.network.lookup[i][j] {
                            if let (Some(start), Some(end)) = (
                                self.node_positions.get(&i),
                                self.node_positions.get(&j)
                            ) {
                                let line = Line::new(PlotPoints::new(vec![
                                    [start[0], start[1]],
                                    [end[0], end[1]],
                                ])).width(if weight > 0.5 { 2.0 } else { 1.0 });
                                plot_ui.line(line);
                            }
                        }
                    }
                }

                // Draw nodes and handle interaction
                if let Some(pointer_pos) = input.pointer.latest_pos() {
                    if let Some(plot_pos) = plot_ui.pointer_coordinate() {
                        for (i, pos) in &mut self.node_positions {
                            // Draw the node
                            let points = Points::new(vec![[pos[0], pos[1]]])
                                .radius(5.0)
                                .color(if Some(*i) == self.selected_node {
                                    egui::Color32::RED
                                } else {
                                    egui::Color32::BLUE
                                });
                            plot_ui.points(points);

                            // Check for clicks near this node
                            let dist = ((plot_pos.x - pos[0]).powi(2) + 
                                      (plot_pos.y - pos[1]).powi(2)).sqrt();
                            if dist < 0.1 { // Adjust this threshold as needed
                                if input.pointer.primary_clicked() {
                                    clicked_node = Some(*i);
                                }
                                if input.pointer.primary_down() {
                                    pos[0] = plot_pos.x;
                                    pos[1] = plot_pos.y;
                                }
                            }
                        }
                    }
                }
            });

            // Update selected node only if we clicked near one
            if let Some(node) = clicked_node {
                self.selected_node = Some(node);
            }

            // Show info panel
            if let Some(node) = self.selected_node {
                egui::Window::new("Node Info")
                    .show(ctx, |ui| {
                        if let Some(node_data) = self.network.nodes.get(&node) {
                            ui.label(format!("Node: {}", node_data.data));
                            ui.label(format!("Degree: {}", self.network.degree(node)));
                        }
                    });
            }
        });

        ctx.request_repaint();
    }
}

#[derive(Debug, Clone)]
/// A network that uses a simple adjacency list for edge storage.
/// 
/// # Attributes
/// 
/// * nodes: HashMap<usize, Node>
/// A map of node identifiers to their corresponding Node structs.
/// 
/// * edges: HashMap<usize, Vec<Edge>>
/// A map of node identifiers to their outgoing edges.
pub struct Network {
    nodes: HashMap<usize, Node>,
    edges: HashMap<usize, Vec<Edge>>,
}
impl Network { 
    /// Creates a new Network.
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The new Network.
    pub fn new() -> Self { 
        Network { 
            nodes: HashMap::new(), 
            edges: HashMap::new(), 
        } 
    } 
    
    /// Adds a node to the network.
    /// 
    /// # Parameters
    /// 
    /// * id: usize
    /// The id of the node
    /// 
    /// * data: String
    /// The data associated with the node.
    /// 
    /// # Returns
    /// 
    /// * Self
    /// The updated Network.
    pub fn add_node(&mut self, id: usize, data: String) {
        let node = Node { id, data }; self.nodes.insert(id, node); 
    } 
    
    /// Adds an edge to the network.
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
    /// 
    /// # Returns
    /// 
    /// * ()
    /// The updated Network.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) { 
        let edge = Edge { from, to, weight }; 
        self.edges.entry(from).or_insert(Vec::new()).push(edge); 
    } 
    
    /// Breadth-First Search. 
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node.
    /// 
    /// # Returns
    /// 
    /// * Vec<usize>
    /// The BFS path
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
    
    /// Find shortest path using Dijkstra's algorithm.
    /// 
    /// # Parameters
    /// 
    /// * start: usize
    /// The id of the starting node.
    /// 
    /// * end: usize
    /// The id of the ending node.
    /// 
    /// # Returns
    /// 
    /// * Option<(Vec<usize>, f64)>
    /// The shortest path and its distance.
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

    /// Plots the network using egui.
    /// 
    /// # Parameters
    /// 
    /// * output_file: &str
    /// The path to the output file.
    /// 
    /// # Returns
    /// 
    /// * Result<(), Box<dyn Error>>
    /// The result of the plot operation.   
    pub fn static_plot(&self, output_file: &str) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::new(output_file, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(0f64..800f64, 0f64..600f64)?;

        // Create random positions for all nodes
        let mut node_positions: HashMap<usize, (f64, f64)> = HashMap::new();
        for &id in self.nodes.keys() {
            node_positions.insert(id, (
                fastrand::f64() * 700.0 + 50.0,
                fastrand::f64() * 500.0 + 50.0
            ));
        }

        // Draw edges
        for (from, edges) in &self.edges {
            if let Some(&(from_x, from_y)) = node_positions.get(from) {
                for edge in edges {
                    if let Some(&(to_x, to_y)) = node_positions.get(&edge.to) {
                        let color = if edge.weight > 0.5 { RED } else { BLUE };
                        chart.plotting_area().draw(&PathElement::new(
                            vec![(from_x, from_y), (to_x, to_y)],
                            color.mix(0.5),
                        ))?;
                    }
                }
            }
        }

        // Draw nodes
        for (&id, &(x, y)) in &node_positions {
            chart.plotting_area().draw(&Circle::new(
                (x, y),
                5,
                ShapeStyle::from(&BLACK).filled(),
            ))?;
            
            if let Some(node) = self.nodes.get(&id) {
                chart.plotting_area().draw(&Text::new(
                    node.data.clone(),
                    (x + 10.0, y),
                    ("sans-serif", 15).into_font(),
                ))?;
            }
        }

        Ok(())
    }

    /// Shows the network interactively using egui.
    /// 
    /// # Returns
    /// 
    /// * Result<(), eframe::Error>
    /// The result of the show operation.
    pub fn show_interactive(&self) -> Result<(), eframe::Error> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([800.0, 600.0]),
            ..Default::default()
        };

        eframe::run_native(
            "Network Visualization",
            options,
            Box::new(|_cc| Ok(Box::new(SimpleNetworkVisualizer::new(self))))
        )
    }
}

/// A simple network visualizer using egui.
/// 
/// # Attributes
/// 
/// * network: &'a Network
/// The network to visualize.
/// 
/// * node_positions: HashMap<usize, [f64; 2]>
/// The positions of the nodes.
/// 
/// * selected_node: Option<usize>
/// The id of the selected node.
/// 
/// * dragging: Option<usize>
/// The id of the node being dragged.   
struct SimpleNetworkVisualizer<'a> {
    network: &'a Network,
    node_positions: HashMap<usize, [f64; 2]>,
    selected_node: Option<usize>,
    dragging: Option<usize>,
}

impl<'a> SimpleNetworkVisualizer<'a> {
    fn new(network: &'a Network) -> Self {
        let mut node_positions = HashMap::new();
        for &id in network.nodes.keys() {
            node_positions.insert(id, [
                fastrand::f64() * 700.0 + 50.0,
                fastrand::f64() * 500.0 + 50.0
            ]);
        }
        Self {
            network,
            node_positions,
            selected_node: None,
            dragging: None,
        }
    }
}

impl<'a> eframe::App for SimpleNetworkVisualizer<'a> {
    /// Updates the network visualizer.
    /// 
    /// # Parameters
    /// 
    /// * ctx: &egui::Context
    /// The context of the application.
    /// 
    /// * _frame: &mut eframe::Frame
    /// The frame of the application.
    /// 
    /// # Returns
    /// 
    /// * ()
    /// The updated network visualizer.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let input = ui.input(|i| i.clone());
            
            let plot = Plot::new("network_plot")
                .allow_drag(false)
                .allow_zoom(true)
                .include_y(0.0)
                .include_y(600.0)
                .include_x(0.0)
                .include_x(800.0);

            let mut clicked_node = None;

            plot.show(ui, |plot_ui| {
                // Draw edges
                for (from, edges) in &self.network.edges {
                    if let Some(&[from_x, from_y]) = self.node_positions.get(from) {
                        for edge in edges {
                            if let Some(&[to_x, to_y]) = self.node_positions.get(&edge.to) {
                                let line = Line::new(PlotPoints::new(vec![
                                    [from_x, from_y],
                                    [to_x, to_y],
                                ])).width(if edge.weight > 0.5 { 2.0 } else { 1.0 });
                                plot_ui.line(line);
                            }
                        }
                    }
                }

                // Draw nodes and handle interaction
                if let Some(pointer_pos) = input.pointer.latest_pos() {
                    if let Some(plot_pos) = plot_ui.pointer_coordinate() {
                        for (i, pos) in &mut self.node_positions {
                            let points = Points::new(vec![[pos[0], pos[1]]])
                                .radius(5.0)
                                .color(if Some(*i) == self.selected_node {
                                    egui::Color32::RED
                                } else {
                                    egui::Color32::BLUE
                                });
                            plot_ui.points(points);

                            let dist = ((plot_pos.x - pos[0]).powi(2) + 
                                      (plot_pos.y - pos[1]).powi(2)).sqrt();
                            if dist < 0.1 {
                                if input.pointer.primary_clicked() {
                                    clicked_node = Some(*i);
                                }
                                if input.pointer.primary_down() {
                                    pos[0] = plot_pos.x;
                                    pos[1] = plot_pos.y;
                                }
                            }
                        }
                    }
                }
            });

            if let Some(node) = clicked_node {
                self.selected_node = Some(node);
            }

            if let Some(node) = self.selected_node {
                egui::Window::new("Node Info")
                    .show(ctx, |ui| {
                        if let Some(node_data) = self.network.nodes.get(&node) {
                            ui.label(format!("Node: {}", node_data.data));
                            let degree = self.network.edges.get(&node)
                                .map(|edges| edges.len())
                                .unwrap_or(0);
                            ui.label(format!("Degree: {}", degree));
                        }
                    });
            }
        });

        ctx.request_repaint();
    } 
} 
    



/// Example usage of the Network struct.
/// 
/// # Returns
/// 
/// * ()
/// The example usage of the Network struct.    
pub fn example() {

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

    //# Uncomment to see the network plots#
    // Plot the network
    //network.static_plot("temp/hybrid_network.png").unwrap();

    // Show interactive plot
    //network.show_interactive().unwrap();

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


    //# Uncomment to see the network plots#
    // Plot the network
    //network.static_plot("temp/network.png").unwrap();
    // Show interactive plot
    //network.show_interactive().unwrap();
}