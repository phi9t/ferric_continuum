use parameter_passing::{
    borrow_large_rect, compute_area_by_ref, compute_area_by_value, consume_large_rect,
    scale_by_mut_ref, transform_by_value, LargeRectangle, Rectangle,
};
use tracing::{info, Level};
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Parameter Passing Demo (Rust) ===");

    let rect = Rectangle::new(10.0, 5.0);
    info!(original = rect.to_string(), "Starting rectangle");

    // 1. Pass by immutable reference - most common
    info!("1. Pass by immutable reference:");
    let area = compute_area_by_ref(&rect);
    info!(area, "Area calculated");
    info!(rect = rect.to_string(), "Original unchanged");
    info!("Use case: Read-only access (most common in Rust)");

    // 2. Pass by value - copies for Copy types
    info!("2. Pass by value (Copy trait):");
    let area = compute_area_by_value(rect);
    info!(area, "Computed area (width doubled internally)");
    info!(rect = rect.to_string(), "Original unchanged (Copy trait)");
    info!("Use case: Small types that implement Copy");

    // 3. Pass by mutable reference - modify in place
    info!("3. Pass by mutable reference:");
    let mut rect_mut = Rectangle::new(10.0, 5.0);
    info!(before = rect_mut.to_string(), "Before modification");
    scale_by_mut_ref(&mut rect_mut, 2.0);
    info!(after = rect_mut.to_string(), "After modification");
    info!("Use case: Modify the original value");

    // 4. Pass by value and transform
    info!("4. Pass by value and transform:");
    let rect2 = Rectangle::new(5.0, 3.0);
    info!(before = rect2.to_string(), "Before transform");
    let result = transform_by_value(rect2, 3.0);
    info!(result = result.to_string(), "After transform");
    info!("Use case: Transform and return new value");

    // 5. Move semantics with non-Copy types
    info!("5. Move semantics (non-Copy type):");
    let large = LargeRectangle::new(10.0, 5.0);

    // Borrow - still usable after
    info!("Borrowing...");
    let area = borrow_large_rect(&large);
    info!(area, "Area from borrow");
    info!("large is still accessible after borrow");

    // Consume - NOT usable after
    info!("Consuming (move)...");
    let area = consume_large_rect(large);
    info!(area, "Area from consume");
    info!("large is NO LONGER accessible (moved)");

    info!("Key Differences from C++:");
    info!("- Rust: Borrow by default (&), move is explicit or automatic");
    info!("- C++: Copy by default, move needs std::move()");
    info!("- Rust enforces borrowing rules at compile time");
    info!("- No null pointers in safe Rust (use Option<&T> instead)");

    info!("Rust Guidelines:");
    info!("- &T      : Default for read-only (most common)");
    info!("- &mut T  : When you need to modify");
    info!("- T       : Takes ownership (move for non-Copy, copy for Copy)");
    info!("- No need for pointer/reference choices - compiler enforces safety!");
}
