use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceTopology {
    pub name: String,
    pub num_qubits: usize,
    pub edges: Vec<(usize, usize)>,
    pub qubit_fidelities: HashMap<usize, f64>,
    pub edge_fidelities: HashMap<(usize, usize), f64>,
}

impl DeviceTopology {
    pub fn new(name: &str, num_qubits: usize) -> Self {
        Self {
            name: name.to_string(),
            num_qubits,
            edges: Vec::new(),
            qubit_fidelities: HashMap::new(),
            edge_fidelities: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, q0: usize, q1: usize, fidelity: f64) {
        self.edges.push((q0, q1));
        self.edge_fidelities.insert((q0, q1), fidelity);
        self.edge_fidelities.insert((q1, q0), fidelity);
    }

    pub fn are_connected(&self, q0: usize, q1: usize) -> bool {
        self.edges.iter().any(|&(a, b)| (a == q0 && b == q1) || (a == q1 && b == q0))
    }

    pub fn neighbors(&self, q: usize) -> Vec<usize> {
        let mut result = Vec::new();
        for &(a, b) in &self.edges {
            if a == q { result.push(b); }
            else if b == q { result.push(a); }
        }
        result
    }

    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to { return Some(vec![from]); }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<usize, usize> = HashMap::new();
        visited.insert(from);
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            for &neighbor in &self.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    if neighbor == to {
                        let mut path = vec![to];
                        let mut curr = to;
                        while let Some(&p) = parent.get(&curr) {
                            path.push(p);
                            curr = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(neighbor);
                }
            }
        }
        None
    }

    pub fn swap_distance(&self, from: usize, to: usize) -> Option<usize> {
        self.shortest_path(from, to).map(|p| if p.len() > 1 { p.len() - 2 } else { 0 })
    }

    pub fn linear(n: usize) -> Self {
        let mut topo = Self::new("linear", n);
        for i in 0..n.saturating_sub(1) {
            topo.add_edge(i, i + 1, 0.99);
        }
        topo
    }

    pub fn grid(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let mut topo = Self::new("grid", n);
        for r in 0..rows {
            for c in 0..cols {
                let q = r * cols + c;
                if c + 1 < cols { topo.add_edge(q, q + 1, 0.99); }
                if r + 1 < rows { topo.add_edge(q, q + cols, 0.99); }
            }
        }
        topo
    }

    /// Heavy-hex lattice (IBM topology): grid with extra bridge qubits.
    pub fn heavy_hex(n: usize) -> Self {
        // Simplified heavy-hex: linear backbone with additional connectivity
        let mut topo = Self::new("heavy_hex", n);
        for i in 0..n.saturating_sub(1) {
            topo.add_edge(i, i + 1, 0.995);
        }
        // Add cross-links every 4 qubits (characteristic of heavy-hex)
        let mut i = 0;
        while i + 3 < n {
            topo.add_edge(i, i + 3, 0.99);
            i += 4;
        }
        topo
    }

    /// All-to-all connectivity (trapped-ion systems).
    pub fn all_to_all(n: usize) -> Self {
        let mut topo = Self::new("all_to_all", n);
        for i in 0..n {
            for j in (i + 1)..n {
                topo.add_edge(i, j, 0.98);
            }
        }
        topo
    }

    /// Binary tree topology.
    pub fn tree(n: usize) -> Self {
        let mut topo = Self::new("tree", n);
        for i in 0..n {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < n { topo.add_edge(i, left, 0.99); }
            if right < n { topo.add_edge(i, right, 0.99); }
        }
        topo
    }

    /// Custom topology from an adjacency list.
    pub fn custom(name: &str, adjacency: &[(usize, usize)], fidelity: f64) -> Self {
        let max_q = adjacency.iter()
            .flat_map(|&(a, b)| [a, b])
            .max()
            .map_or(0, |m| m + 1);
        let mut topo = Self::new(name, max_q);
        for &(a, b) in adjacency {
            topo.add_edge(a, b, fidelity);
        }
        topo
    }

    /// Average connectivity (edges per qubit).
    pub fn avg_connectivity(&self) -> f64 {
        if self.num_qubits == 0 { return 0.0; }
        (2.0 * self.edges.len() as f64) / self.num_qubits as f64
    }

    /// Diameter of the graph (longest shortest path).
    pub fn diameter(&self) -> usize {
        let mut max_dist = 0;
        for i in 0..self.num_qubits {
            for j in (i + 1)..self.num_qubits {
                if let Some(path) = self.shortest_path(i, j) {
                    let dist = path.len().saturating_sub(1);
                    if dist > max_dist { max_dist = dist; }
                }
            }
        }
        max_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology() {
        let topo = DeviceTopology::linear(5);
        assert!(topo.are_connected(0, 1));
        assert!(!topo.are_connected(0, 2));
        assert_eq!(topo.swap_distance(0, 4), Some(3));
    }

    #[test]
    fn test_grid_topology() {
        let topo = DeviceTopology::grid(3, 3);
        assert_eq!(topo.num_qubits, 9);
        assert!(topo.are_connected(0, 1));
        assert!(topo.are_connected(0, 3));
        assert!(!topo.are_connected(0, 4));
    }
}
