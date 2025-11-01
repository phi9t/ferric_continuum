use tracing::{info, Level};
use tracing_subscriber;
use value_semantics::{Data, Point};

fn main() {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Value Semantics Demo (Rust) ===");

    // Point implements Copy - value semantics
    let p1 = Point::new(3.0, 4.0);
    info!(point = p1.to_string(), "p1 created");
    info!(distance = %p1.distance_from_origin(), "Distance from origin");

    // Copy creates independent value
    let p2 = p1; // Copy happens automatically
    info!(point = p2.to_string(), "p2 copied from p1");

    // Translate p2 - p1 remains unchanged
    let p2 = p2.translate(2.0, 1.0);
    info!("After translating p2:");
    info!(p1 = p1.to_string(), "p1 unchanged");
    info!(p2 = p2.to_string(), "p2 translated");

    info!("--- Contrast with Move Semantics ---");

    // Data doesn't implement Copy - move semantics
    let d1 = Data::new(vec![1, 2, 3, 4, 5]);
    info!(len = d1.len(), sum = d1.sum(), "d1 created");

    let d2 = d1; // Move! d1 is no longer accessible
    info!(len = d2.len(), sum = d2.sum(), "d2 moved from d1");
    info!("d1 is no longer accessible (moved)");

    info!("Key Differences:");
    info!("- C++: Copy by default (value semantics)");
    info!("- Rust: Move by default, Copy must be explicit");
    info!("- Both approaches prevent accidental sharing");
}
