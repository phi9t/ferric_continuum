/// Move Semantics in Rust
///
/// In Rust, move semantics are the DEFAULT for most types.
/// Values are moved unless the type implements Copy.
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A type that owns heap data - moves by default
pub struct LargeBuffer {
    data: Vec<i32>,
}

impl LargeBuffer {
    pub fn new(size: usize) -> Self {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        LargeBuffer {
            data: vec![0; size],
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn fill(&mut self, value: i32) {
        for item in &mut self.data {
            *item = value;
        }
    }

    pub fn alloc_count() -> usize {
        ALLOC_COUNT.load(Ordering::SeqCst)
    }

    pub fn dealloc_count() -> usize {
        DEALLOC_COUNT.load(Ordering::SeqCst)
    }

    pub fn reset_counts() {
        ALLOC_COUNT.store(0, Ordering::SeqCst);
        DEALLOC_COUNT.store(0, Ordering::SeqCst);
    }
}

impl Drop for LargeBuffer {
    fn drop(&mut self) {
        DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
    }
}

// Explicit clone for when you actually need a copy
impl Clone for LargeBuffer {
    fn clone(&self) -> Self {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        LargeBuffer {
            data: self.data.clone(),
        }
    }
}

/// Returns by value - ownership transferred to caller (move)
pub fn create_buffer(size: usize) -> LargeBuffer {
    let mut buf = LargeBuffer::new(size);
    buf.fill(42);
    buf // Ownership moved to caller
}

/// Takes ownership by value - original can't be used after call
pub fn process_buffer(mut buf: LargeBuffer) -> LargeBuffer {
    buf.fill(100);
    buf // Ownership moved back to caller
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_semantics() {
        LargeBuffer::reset_counts();

        let buf1 = create_buffer(1000);
        // Only 1 allocation - no copies!
        assert_eq!(LargeBuffer::alloc_count(), 1);

        // Move buf1 into process_buffer
        let buf2 = process_buffer(buf1);
        // Still only 1 allocation (moved, not copied)
        assert_eq!(LargeBuffer::alloc_count(), 1);

        // buf1 is no longer accessible here (moved)
        // This would fail to compile:
        // println!("buf1 size: {}", buf1.size());

        assert_eq!(buf2.size(), 1000);
    }

    #[test]
    fn test_explicit_clone() {
        LargeBuffer::reset_counts();

        let buf1 = LargeBuffer::new(1000);
        assert_eq!(LargeBuffer::alloc_count(), 1);

        // Explicit clone creates a copy
        let buf2 = buf1.clone();
        assert_eq!(LargeBuffer::alloc_count(), 2);

        // Both are accessible
        assert_eq!(buf1.size(), 1000);
        assert_eq!(buf2.size(), 1000);
    }

    #[test]
    fn test_drop_on_scope_exit() {
        LargeBuffer::reset_counts();

        {
            let _buf = LargeBuffer::new(100);
            assert_eq!(LargeBuffer::dealloc_count(), 0);
        } // buf dropped here

        assert_eq!(LargeBuffer::dealloc_count(), 1);
    }
}
