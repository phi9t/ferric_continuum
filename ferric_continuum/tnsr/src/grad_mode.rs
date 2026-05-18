use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

pub fn set_enabled(enabled: bool) {
    GRAD_ENABLED.with(|c| c.set(enabled));
}

pub fn is_enabled_and_any_requires_grad(tensors: &[&crate::tensor::Tensor]) -> bool {
    is_enabled()
        && tensors
            .iter()
            .any(|t| t.inner.borrow().autograd.requires_grad)
}

pub struct NoGradGuard {
    prev: bool,
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = is_enabled();
        set_enabled(false);
        Self { prev }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_enabled(self.prev);
    }
}
