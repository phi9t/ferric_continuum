use smart_pointers::{
    count_nodes, create_list, share_resource, Counter, FileGuard, Resource,
};
use std::rc::Rc;
use tracing::{info, Level};
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("=== Smart Pointers & RAII Demo (Rust) ===");

    // ===== Box - Exclusive Ownership =====
    info!("1. Box<T> - Exclusive Ownership");
    {
        let list = create_list(&[1, 2, 3, 4, 5]);
        let node_count = count_nodes(list.as_ref().map(|b| b.as_ref()));
        info!(node_count, "Created list");

        if let Some(ref node) = list {
            info!(value = node.value(), "First value");
        }

        // Transfer ownership
        let list2 = list; // list moved
        info!("After move:");
        info!("- list is moved (no longer accessible)");
        info!(owns_data = list2.is_some(), "list2 owns the data");

        info!("Leaving scope - automatic cleanup!");
    } // list2 dropped, all nodes automatically freed
    info!("All memory cleaned up automatically.");

    // ===== Rc - Shared Ownership =====
    info!("2. Rc<T> - Shared Ownership");
    {
        Resource::reset_count();
        info!(alive = Resource::instance_count(), "Resources alive");

        let resource = Resource::new(42);
        info!(id = resource.id(), "Created resource");
        info!(count = Rc::strong_count(&resource), "Strong count");
        info!(alive = Resource::instance_count(), "Resources alive");

        {
            // Share ownership
            let _shared = share_resource(Rc::clone(&resource), 3);
            info!("After sharing with 3 more owners:");
            info!(count = Rc::strong_count(&resource), "Strong count");
            info!(alive = Resource::instance_count(), "Resources alive");

            info!("Leaving inner scope...");
        } // _shared vector dropped

        info!("Back to outer scope:");
        info!(count = Rc::strong_count(&resource), "Strong count");
        info!(alive = Resource::instance_count(), "Resources alive");

        info!("Leaving outer scope...");
    } // resource dropped when last Rc goes away
    info!(alive = Resource::instance_count(), "Resources alive");

    // ===== RAII Pattern =====
    info!("3. RAII Pattern (Drop trait)");
    {
        let file = FileGuard::new("data.txt");
        info!(filename = file.filename(), "File opened");
        info!(is_open = file.is_open(), "File status");

        // Transfer ownership with move
        let file2 = file; // file moved to file2
        info!("After move:");
        info!(owns_resource = file2.is_open(), "file2 owns resource");

        info!("Leaving scope...");
    } // file2 dropped, Drop::drop called automatically
    info!("File automatically closed in Drop::drop.");

    // ===== Interior Mutability =====
    info!("4. Interior Mutability (RefCell)");
    {
        let counter = Counter::new();
        info!(value = counter.get(), "Initial value");

        // Multiple references can mutate through RefCell
        let c1 = Rc::clone(&counter);
        let c2 = Rc::clone(&counter);

        c1.increment();
        c2.increment();
        counter.increment();

        info!(value = counter.get(), "After 3 increments");
        info!(count = Rc::strong_count(&counter), "Strong count");
    }

    info!("Key Differences:");
    info!("- C++: Manual smart pointer selection (unique_ptr vs shared_ptr)");
    info!("- Rust: Ownership enforced at compile time");
    info!("- Box<T>: Like unique_ptr (exclusive ownership)");
    info!("- Rc<T>: Like shared_ptr (reference counted)");
    info!("- Arc<T>: Thread-safe Rc (atomic reference counting)");
    info!("- Drop trait: Automatic cleanup (like RAII/destructors)");
    info!("- RefCell: Interior mutability when needed");

    info!("Rust's Advantages:");
    info!("- Compiler prevents use-after-free at compile time");
    info!("- No null pointer dereferencing in safe code");
    info!("- Memory leaks harder to create (but still possible with Rc cycles)");
}
