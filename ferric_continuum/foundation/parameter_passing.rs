/// Parameter Passing in Rust
///
/// Rust's borrowing rules enforce memory safety at compile time.
/// You can pass by value (move), by reference (&), or by mutable reference (&mut).

#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    pub width: f64,
    pub height: f64,
}

impl Rectangle {
    pub fn new(width: f64, height: f64) -> Self {
        Rectangle { width, height }
    }

    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    pub fn to_string(&self) -> String {
        format!("Rectangle({:.2} x {:.2})", self.width, self.height)
    }
}

// =============================================================================
// Different Parameter Passing Strategies
// =============================================================================

/// 1. Pass by immutable reference - read-only access, no ownership transfer
/// Most common: Equivalent to C++ const&
pub fn compute_area_by_ref(rect: &Rectangle) -> f64 {
    rect.area()
}

/// 2. Pass by value - takes ownership (or copies if type implements Copy)
/// For Copy types like Rectangle: creates a copy
/// For non-Copy types: moves ownership
pub fn compute_area_by_value(mut rect: Rectangle) -> f64 {
    rect.width *= 2.0; // Local modification
    rect.area()
}

/// 3. Pass by mutable reference - allows modification without taking ownership
/// Equivalent to C++ non-const&
pub fn scale_by_mut_ref(rect: &mut Rectangle, factor: f64) {
    rect.width *= factor;
    rect.height *= factor;
}

/// 4. Pass by value and consume - takes ownership
/// For non-Copy types, this prevents caller from using the value again
pub fn transform_by_value(mut rect: Rectangle, scale: f64) -> Rectangle {
    rect.width *= scale;
    rect.height *= scale;
    rect
}

/// Example with non-Copy type to show move semantics
#[derive(Debug)]
pub struct LargeRectangle {
    width: f64,
    height: f64,
    data: Vec<i32>, // Makes it non-Copy
}

impl LargeRectangle {
    pub fn new(width: f64, height: f64) -> Self {
        LargeRectangle {
            width,
            height,
            data: vec![0; 1000],
        }
    }

    pub fn area(&self) -> f64 {
        self.width * self.height
    }
}

/// Takes ownership - caller can't use the value after this
pub fn consume_large_rect(rect: LargeRectangle) -> f64 {
    rect.area()
}

/// Borrows - caller can still use the value after this
pub fn borrow_large_rect(rect: &LargeRectangle) -> f64 {
    rect.area()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_by_reference() {
        let rect = Rectangle::new(10.0, 5.0);
        let area = compute_area_by_ref(&rect);

        assert_eq!(area, 50.0);
        // rect is still accessible
        assert_eq!(rect.width, 10.0);
    }

    #[test]
    fn test_by_value_copy() {
        let rect = Rectangle::new(10.0, 5.0);
        let area = compute_area_by_value(rect);

        // rect is copied (implements Copy), original unchanged
        assert_eq!(rect.width, 10.0);
        assert_eq!(area, 100.0); // Width was doubled inside function
    }

    #[test]
    fn test_by_mut_reference() {
        let mut rect = Rectangle::new(10.0, 5.0);
        scale_by_mut_ref(&mut rect, 2.0);

        assert_eq!(rect.width, 20.0);
        assert_eq!(rect.height, 10.0);
    }

    #[test]
    fn test_move_semantics() {
        let large = LargeRectangle::new(10.0, 5.0);

        // Borrow - still accessible after
        let area = borrow_large_rect(&large);
        assert_eq!(area, 50.0);
        assert_eq!(large.width, 10.0); // Still accessible

        // Consume - NOT accessible after
        let area2 = consume_large_rect(large);
        assert_eq!(area2, 50.0);
        // large is no longer accessible here (moved)
        // assert_eq!(large.width, 10.0);  // Would fail to compile!
    }

    #[test]
    fn test_mutable_borrow_rules() {
        let mut rect = Rectangle::new(10.0, 5.0);

        // Can have multiple immutable borrows
        let _ref1 = &rect;
        let _ref2 = &rect;
        // All OK!

        // But can't have mutable borrow while immutable borrows exist
        // let _mut_ref = &mut rect;  // Would fail to compile!

        // After immutable borrows go out of scope, mutable borrow is OK
        scale_by_mut_ref(&mut rect, 2.0);
        assert_eq!(rect.width, 20.0);
    }
}
