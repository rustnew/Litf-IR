use serde::{Serialize, Deserialize};

/// A complex number represented as (real, imaginary).
pub type Complex = (f64, f64);

/// A complex matrix stored as a flat Vec of (real, imag) tuples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Complex>,
}

impl ComplexMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![(0.0, 0.0); rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.data[i * n + i] = (1.0, 0.0);
        }
        m
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Complex {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, val: Complex) {
        self.data[r * self.cols + c] = val;
    }

    /// Conjugate transpose (dagger).
    pub fn dagger(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let (re, im) = self.get(r, c);
                result.set(c, r, (re, -im));
            }
        }
        result
    }

    /// Matrix multiply: self * other.
    pub fn mul(&self, other: &Self) -> Option<Self> {
        if self.cols != other.rows { return None; }
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = (0.0, 0.0);
                for k in 0..self.cols {
                    let (ar, ai) = self.get(i, k);
                    let (br, bi) = other.get(k, j);
                    sum.0 += ar * br - ai * bi;
                    sum.1 += ar * bi + ai * br;
                }
                result.set(i, j, sum);
            }
        }
        Some(result)
    }

    /// Trace of a square matrix.
    pub fn trace(&self) -> Option<Complex> {
        if self.rows != self.cols { return None; }
        let mut sum = (0.0, 0.0);
        for i in 0..self.rows {
            let (re, im) = self.get(i, i);
            sum.0 += re;
            sum.1 += im;
        }
        Some(sum)
    }
}

/// Kraus channel: a quantum channel described by Kraus operators {K_i}
/// such that Sum(K_i† K_i) = I.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KrausChannel {
    pub operators: Vec<ComplexMatrix>,
    pub dimension: usize,
}

impl KrausChannel {
    /// Create a new Kraus channel from a set of operators.
    /// Returns None if the operators are empty or have inconsistent dimensions.
    pub fn new(operators: Vec<ComplexMatrix>) -> Option<Self> {
        if operators.is_empty() { return None; }
        let dim = operators[0].rows;
        if operators.iter().any(|op| op.rows != dim || op.cols != dim) {
            return None;
        }
        Some(Self { operators, dimension: dim })
    }

    /// Compose two channels: apply `self` first, then `other`.
    pub fn compose(&self, other: &KrausChannel) -> Option<KrausChannel> {
        if self.dimension != other.dimension { return None; }
        let mut new_ops = Vec::with_capacity(self.operators.len() * other.operators.len());
        for ki in &other.operators {
            for kj in &self.operators {
                if let Some(prod) = ki.mul(kj) {
                    new_ops.push(prod);
                }
            }
        }
        KrausChannel::new(new_ops)
    }

    /// Average gate fidelity of this channel relative to identity.
    /// F_avg = (d * F_e + 1) / (d + 1), where F_e = entanglement fidelity.
    pub fn average_gate_fidelity(&self) -> f64 {
        let d = self.dimension as f64;
        let identity = ComplexMatrix::identity(self.dimension);
        let mut f_e = 0.0;
        for k in &self.operators {
            if let Some(prod) = identity.dagger().mul(k) {
                if let Some(tr) = prod.trace() {
                    f_e += tr.0 * tr.0 + tr.1 * tr.1; // |Tr(K_i)|^2
                }
            }
        }
        f_e /= d * d;
        (d * f_e + 1.0) / (d + 1.0)
    }

    /// Depolarising channel: ρ → (1-p)*ρ + (p/d)*I
    pub fn depolarizing(p: f64, n_qubits: u32) -> Self {
        let d = 1usize << n_qubits;
        let sqrt_main = ((1.0 - p) as f64).sqrt();
        let mut main = ComplexMatrix::identity(d);
        for i in 0..d {
            main.set(i, i, (sqrt_main, 0.0));
        }

        let sqrt_dep = (p / (d * d) as f64).sqrt();
        let mut ops = vec![main];
        // For a single qubit, use Pauli operators
        if n_qubits == 1 {
            // sigma_x
            let mut sx = ComplexMatrix::new(2, 2);
            sx.set(0, 1, (sqrt_dep, 0.0));
            sx.set(1, 0, (sqrt_dep, 0.0));
            ops.push(sx);
            // sigma_y
            let mut sy = ComplexMatrix::new(2, 2);
            sy.set(0, 1, (0.0, -sqrt_dep));
            sy.set(1, 0, (0.0, sqrt_dep));
            ops.push(sy);
            // sigma_z
            let mut sz = ComplexMatrix::new(2, 2);
            sz.set(0, 0, (sqrt_dep, 0.0));
            sz.set(1, 1, (-sqrt_dep, 0.0));
            ops.push(sz);
        }
        Self { operators: ops, dimension: d }
    }

    /// Amplitude damping channel (T1 process): γ = 1 - e^(-t/T1)
    pub fn amplitude_damping(gamma: f64) -> Self {
        let mut k0 = ComplexMatrix::new(2, 2);
        k0.set(0, 0, (1.0, 0.0));
        k0.set(1, 1, ((1.0 - gamma).sqrt(), 0.0));

        let mut k1 = ComplexMatrix::new(2, 2);
        k1.set(0, 1, (gamma.sqrt(), 0.0));

        Self { operators: vec![k0, k1], dimension: 2 }
    }

    /// Phase damping channel (T2 dephasing): λ = 1 - e^(-t/T_φ)
    pub fn phase_damping(lambda: f64) -> Self {
        let mut k0 = ComplexMatrix::new(2, 2);
        k0.set(0, 0, (1.0, 0.0));
        k0.set(1, 1, ((1.0 - lambda).sqrt(), 0.0));

        let mut k1 = ComplexMatrix::new(2, 2);
        k1.set(1, 1, (lambda.sqrt(), 0.0));

        Self { operators: vec![k0, k1], dimension: 2 }
    }

    /// Pauli channel: ρ → (1-px-py-pz)*ρ + px*X*ρ*X + py*Y*ρ*Y + pz*Z*ρ*Z
    pub fn pauli(px: f64, py: f64, pz: f64) -> Self {
        let p_id = 1.0 - px - py - pz;
        let mut k_id = ComplexMatrix::identity(2);
        for i in 0..2 { k_id.set(i, i, (p_id.sqrt(), 0.0)); }

        let sqpx = px.sqrt();
        let mut kx = ComplexMatrix::new(2, 2);
        kx.set(0, 1, (sqpx, 0.0));
        kx.set(1, 0, (sqpx, 0.0));

        let sqpy = py.sqrt();
        let mut ky = ComplexMatrix::new(2, 2);
        ky.set(0, 1, (0.0, -sqpy));
        ky.set(1, 0, (0.0, sqpy));

        let sqpz = pz.sqrt();
        let mut kz = ComplexMatrix::new(2, 2);
        kz.set(0, 0, (sqpz, 0.0));
        kz.set(1, 1, (-sqpz, 0.0));

        Self { operators: vec![k_id, kx, ky, kz], dimension: 2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_channel_fidelity() {
        let id = KrausChannel::new(vec![ComplexMatrix::identity(2)]);
        assert!(id.is_some());
        let ch = id.unwrap();
        let f = ch.average_gate_fidelity();
        assert!((f - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_depolarizing_fidelity() {
        let ch = KrausChannel::depolarizing(0.01, 1);
        let f = ch.average_gate_fidelity();
        assert!(f > 0.99 && f <= 1.0);
    }

    #[test]
    fn test_amplitude_damping() {
        let ch = KrausChannel::amplitude_damping(0.05);
        let f = ch.average_gate_fidelity();
        assert!(f > 0.9 && f <= 1.0);
    }

    #[test]
    fn test_phase_damping() {
        let ch = KrausChannel::phase_damping(0.03);
        let f = ch.average_gate_fidelity();
        assert!(f > 0.9 && f <= 1.0);
    }

    #[test]
    fn test_pauli_channel() {
        let ch = KrausChannel::pauli(0.01, 0.01, 0.01);
        let f = ch.average_gate_fidelity();
        assert!(f > 0.9 && f <= 1.0);
    }

    #[test]
    fn test_compose_channels() {
        let c1 = KrausChannel::depolarizing(0.01, 1);
        let c2 = KrausChannel::amplitude_damping(0.02);
        let composed = c1.compose(&c2);
        assert!(composed.is_some());
        let f = composed.unwrap().average_gate_fidelity();
        assert!(f > 0.9 && f <= 1.0);
    }

    #[test]
    fn test_complex_matrix_dagger() {
        let mut m = ComplexMatrix::new(2, 2);
        m.set(0, 0, (1.0, 0.0));
        m.set(0, 1, (0.0, 1.0));
        m.set(1, 0, (0.0, -1.0));
        m.set(1, 1, (1.0, 0.0));
        let d = m.dagger();
        assert_eq!(d.get(0, 0), (1.0, 0.0));
        assert_eq!(d.get(0, 1), (0.0, 1.0));
        assert_eq!(d.get(1, 0), (0.0, -1.0));
        assert_eq!(d.get(1, 1), (1.0, 0.0));
    }
}
