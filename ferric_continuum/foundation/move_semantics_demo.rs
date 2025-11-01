use move_semantics::{create_buffer, process_buffer, LargeBuffer};
use tracing::{info, Level};
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Move Semantics Demo (Rust) ===");
    info!("In Rust, moves are the DEFAULT behavior");

    // Scenario 1: Create and return (move)
    info!("1. Creating buffer (ownership transferred)...");
    LargeBuffer::reset_counts();
    let buf1 = create_buffer(10000);
    info!(allocations = LargeBuffer::alloc_count(), "After creation");
    info!("Result: Only 1 allocation - no copies!");

    // Scenario 2: Move into function
    info!("2. Moving buffer into function...");
    LargeBuffer::reset_counts();
    let buf2 = create_buffer(10000);
    info!(allocations = LargeBuffer::alloc_count(), "Before process");

    let buf3 = process_buffer(buf2); // buf2 moved
    info!(allocations = LargeBuffer::alloc_count(), "After process");
    info!("Result: Still only 1 allocation (moved, not copied)!");

    // buf2 is no longer accessible here
    info!("buf2 is no longer accessible (moved)");

    // Scenario 3: Explicit clone when you need a copy
    info!("3. Explicit clone (when you need both values)...");
    LargeBuffer::reset_counts();
    let buf4 = LargeBuffer::new(10000);
    info!(allocations = LargeBuffer::alloc_count(), "After creation");

    let buf5 = buf4.clone(); // Explicit copy
    info!(allocations = LargeBuffer::alloc_count(), "After clone");
    info!("Result: 2 allocations - explicit copy made.");

    // Both buf4 and buf5 are accessible
    info!(buf4_size = buf4.size(), "buf4 accessible");
    info!(buf5_size = buf5.size(), "buf5 accessible");

    // Scenario 4: Automatic cleanup
    info!("4. Automatic cleanup with Drop...");
    LargeBuffer::reset_counts();
    {
        let _buf = LargeBuffer::new(1000);
        info!(deallocations = LargeBuffer::dealloc_count(), "Inside scope");
    } // _buf dropped here
    info!(deallocations = LargeBuffer::dealloc_count(), "After scope");
    info!("Result: Automatic cleanup when going out of scope!");

    info!("Key Differences:");
    info!("- C++: Copy by default, move is opt-in with std::move()");
    info!("- Rust: Move by default, copy is opt-in with .clone()");
    info!("- Rust enforces move at compile time (prevents use-after-move)");
    info!("- Both enable efficient resource management");
}
