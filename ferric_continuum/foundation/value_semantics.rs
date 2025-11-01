/// Value Semantics in Rust
///
/// Rust has different default behavior: types are MOVED by default, not copied.
/// To get value semantics (copy behavior), types must implement the Copy trait.

/// Point with Copy trait - behaves like C++ value semantics
#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    pub fn x(&self) -> f64 {
        self.x
    }
    pub fn y(&self) -> f64 {
        self.y
    }

    pub fn translate(&self, dx: f64, dy: f64) -> Point {
        Point::new(self.x + dx, self.y + dy)
    }

    pub fn distance_from_origin(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn to_string(&self) -> String {
        format!("Point({:.2}, {:.2})", self.x, self.y)
    }
}

/// Type WITHOUT Copy - demonstrates move semantics (Rust's default)
#[derive(Debug, Clone)]
pub struct Data {
    values: Vec<i32>,
}

impl Data {
    pub fn new(values: Vec<i32>) -> Self {
        Data { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn sum(&self) -> i32 {
        self.values.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_semantics() {
        let p1 = Point::new(3.0, 4.0);
        let p2 = p1; // Copies because Point implements Copy

        // Both p1 and p2 are usable (value semantics)
        assert_eq!(p1.x(), 3.0);
        assert_eq!(p2.x(), 3.0);
    }

    #[test]
    fn test_independent_copies() {
        let p1 = Point::new(3.0, 4.0);
        let p2 = p1.translate(2.0, 1.0);

        // p1 is unchanged
        assert_eq!(p1.x(), 3.0);
        assert_eq!(p1.y(), 4.0);

        // p2 has new values
        assert_eq!(p2.x(), 5.0);
        assert_eq!(p2.y(), 5.0);
    }

    #[test]
    fn test_move_semantics_without_copy() {
        let d1 = Data::new(vec![1, 2, 3]);
        let d2 = d1; // Moves because Data doesn't implement Copy

        // d1 is no longer accessible here (moved)
        // assert_eq!(d1.len(), 3);  // This would fail to compile!

        // Only d2 is accessible
        assert_eq!(d2.len(), 3);
    }
}
