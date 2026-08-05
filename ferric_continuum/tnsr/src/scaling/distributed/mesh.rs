//! Logical device mesh — the coordinate space parallelism is laid out on.
//!
//! Book reference: Ch.5 "Parallelize a Transformer for Training",
//! <https://jax-ml.github.io/scaling-book/training/> (the device-mesh figures).
//!
//! PyTorch counterpart: [`torch.distributed.device_mesh.DeviceMesh`], which
//! names each axis (`"dp"`, `"tp"`, ...) so a parallelism strategy can be
//! attached to an axis rather than a raw rank range.
//!
//! [`torch.distributed.device_mesh.DeviceMesh`]: https://pytorch.org/docs/stable/distributed.tensor.html
//!
//! `tnsr` never launches these devices; the mesh is pure bookkeeping that the
//! cost estimators consult for "how many devices along this axis?".

/// A named, multi-axis grid of logical devices.
///
/// Devices are numbered `0..total_devices()` in **row-major** order over
/// `shape` (the last axis varies fastest), matching PyTorch/NumPy layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMesh {
    /// Size of each axis, e.g. `[2, 2]` for a 2×2 mesh.
    pub shape: Vec<usize>,
    /// Human-readable name per axis, e.g. `["dp", "tp"]`. Same length as `shape`.
    pub axis_names: Vec<String>,
}

impl DeviceMesh {
    /// A single-axis mesh of `size` devices (e.g. pure data parallelism).
    pub fn new_1d(size: usize, axis: &str) -> Self {
        assert!(size >= 1, "mesh axis must have at least one device");
        DeviceMesh {
            shape: vec![size],
            axis_names: vec![axis.to_string()],
        }
    }

    /// A two-axis mesh, e.g. `new_2d(2, "dp", 2, "tp")` for hybrid DP×TP.
    pub fn new_2d(size0: usize, axis0: &str, size1: usize, axis1: &str) -> Self {
        assert!(
            size0 >= 1 && size1 >= 1,
            "mesh axes must have at least one device"
        );
        DeviceMesh {
            shape: vec![size0, size1],
            axis_names: vec![axis0.to_string(), axis1.to_string()],
        }
    }

    /// Total device count = product of all axis sizes.
    pub fn total_devices(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of devices along the axis named `axis`, or `None` if absent.
    pub fn axis_size(&self, axis: &str) -> Option<usize> {
        self.axis_names
            .iter()
            .position(|a| a == axis)
            .map(|i| self.shape[i])
    }

    /// Row-major coordinates of `rank` within the mesh.
    ///
    /// For a `[2, 2]` mesh: rank 0 → `[0,0]`, rank 1 → `[0,1]`, rank 2 →
    /// `[1,0]`, rank 3 → `[1,1]`.
    pub fn coords(&self, rank: usize) -> Vec<usize> {
        assert!(
            rank < self.total_devices(),
            "rank {rank} out of range for mesh with {} devices",
            self.total_devices()
        );
        let mut coords = vec![0usize; self.shape.len()];
        let mut r = rank;
        // Last axis varies fastest: divide out axis sizes from the right.
        for i in (0..self.shape.len()).rev() {
            coords[i] = r % self.shape[i];
            r /= self.shape[i];
        }
        coords
    }
}
