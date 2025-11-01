use std::cell::RefCell;
/// Smart Pointers in Rust
///
/// Rust has several smart pointer types similar to C++:
/// - Box<T>: Similar to unique_ptr (heap allocation, exclusive ownership)
/// - Rc<T>: Similar to shared_ptr (reference counted, shared ownership)
/// - Arc<T>: Thread-safe version of Rc
///
/// Unlike C++, Rust enforces ownership rules at compile time!
use std::rc::Rc;

// =============================================================================
// Box - Exclusive Ownership (like unique_ptr)
// =============================================================================

pub struct Node {
    value: i32,
    next: Option<Box<Node>>, // Exclusive ownership
}

impl Node {
    pub fn new(value: i32) -> Self {
        Node { value, next: None }
    }

    pub fn value(&self) -> i32 {
        self.value
    }

    pub fn next(&self) -> Option<&Node> {
        self.next.as_ref().map(|boxed| boxed.as_ref())
    }

    pub fn append(&mut self, node: Node) {
        if let Some(ref mut next) = self.next {
            next.append(node);
        } else {
            self.next = Some(Box::new(node));
        }
    }
}

/// Create a linked list with Box (automatic cleanup)
pub fn create_list(values: &[i32]) -> Option<Box<Node>> {
    if values.is_empty() {
        return None;
    }

    let mut head = Box::new(Node::new(values[0]));

    for &value in &values[1..] {
        head.append(Node::new(value));
    }

    Some(head)
}

pub fn count_nodes(head: Option<&Node>) -> usize {
    match head {
        None => 0,
        Some(node) => {
            let mut count = 1;
            let mut current = node;

            while let Some(next) = current.next() {
                count += 1;
                current = next;
            }

            count
        }
    }
}

// =============================================================================
// Rc - Shared Ownership (like shared_ptr)
// =============================================================================

use std::sync::atomic::{AtomicUsize, Ordering};

static RESOURCE_COUNT: AtomicUsize = AtomicUsize::new(0);

pub struct Resource {
    id: i32,
}

impl Resource {
    pub fn new(id: i32) -> Rc<Self> {
        RESOURCE_COUNT.fetch_add(1, Ordering::SeqCst);
        Rc::new(Resource { id })
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn instance_count() -> usize {
        RESOURCE_COUNT.load(Ordering::SeqCst)
    }

    pub fn reset_count() {
        RESOURCE_COUNT.store(0, Ordering::SeqCst);
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        RESOURCE_COUNT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Share ownership - multiple Rc pointers to same resource
pub fn share_resource(resource: Rc<Resource>, copies: usize) -> Vec<Rc<Resource>> {
    (0..copies).map(|_| Rc::clone(&resource)).collect()
}

// =============================================================================
// RAII Pattern (automatic with Drop)
// =============================================================================

pub struct FileGuard {
    filename: String,
    is_open: bool,
}

impl FileGuard {
    pub fn new(filename: &str) -> Self {
        // Simulate opening file
        FileGuard {
            filename: filename.to_string(),
            is_open: true,
        }
    }

    pub fn is_open(&self) -> bool {
        self.is_open
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }
}

impl Drop for FileGuard {
    fn drop(&mut self) {
        if self.is_open {
            // Simulate closing file
            self.is_open = false;
        }
    }
}

// =============================================================================
// Interior Mutability with RefCell
// =============================================================================

/// Demonstrates interior mutability pattern
pub struct Counter {
    value: RefCell<i32>,
}

impl Counter {
    pub fn new() -> Rc<Self> {
        Rc::new(Counter {
            value: RefCell::new(0),
        })
    }

    pub fn increment(&self) {
        *self.value.borrow_mut() += 1;
    }

    pub fn get(&self) -> i32 {
        *self.value.borrow()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_exclusive_ownership() {
        let list = create_list(&[1, 2, 3, 4, 5]);
        assert_eq!(count_nodes(list.as_ref().map(|b| b.as_ref())), 5);

        // Ownership transfer
        let list2 = list; // list moved to list2
                          // list is no longer accessible
        assert!(list2.is_some());
    }

    #[test]
    fn test_rc_shared_ownership() {
        Resource::reset_count();

        let resource = Resource::new(42);
        assert_eq!(Rc::strong_count(&resource), 1);
        assert_eq!(Resource::instance_count(), 1);

        {
            let _shared = share_resource(Rc::clone(&resource), 3);
            assert_eq!(Rc::strong_count(&resource), 4); // Original + 3 clones
            assert_eq!(Resource::instance_count(), 1); // Still only 1 Resource
        } // _shared dropped

        assert_eq!(Rc::strong_count(&resource), 1); // Back to 1
    }

    #[test]
    fn test_automatic_drop() {
        Resource::reset_count();

        {
            let _r1 = Resource::new(1);
            let _r2 = Resource::new(2);
            assert_eq!(Resource::instance_count(), 2);
        } // Both dropped

        assert_eq!(Resource::instance_count(), 0);
    }

    #[test]
    fn test_interior_mutability() {
        let counter = Counter::new();

        let c1 = Rc::clone(&counter);
        let c2 = Rc::clone(&counter);

        c1.increment();
        c2.increment();

        assert_eq!(counter.get(), 2);
    }
}
