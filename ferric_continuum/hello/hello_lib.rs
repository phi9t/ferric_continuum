/// Generate a greeting message
pub fn generate_greeting(name: &str) -> String {
    format!(
        "Hello, {}! Welcome to Ferric Continuum (Rust Edition)",
        name
    )
}

/// Calculate fibonacci number (demonstrating simple algorithm)
pub fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}

/// Check if a number is prime
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }

    let limit = (n as f64).sqrt() as u64;
    for i in (3..=limit).step_by(2) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Get all prime numbers up to n
pub fn primes_up_to(n: u64) -> Vec<u64> {
    (2..=n).filter(|&i| is_prime(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_greeting() {
        assert_eq!(
            generate_greeting("World"),
            "Hello, World! Welcome to Ferric Continuum (Rust Edition)"
        );
        assert_eq!(
            generate_greeting("Ferric"),
            "Hello, Ferric! Welcome to Ferric Continuum (Rust Edition)"
        );
    }

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 1);
        assert_eq!(fibonacci(3), 2);
        assert_eq!(fibonacci(4), 3);
        assert_eq!(fibonacci(5), 5);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(9));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_primes_up_to() {
        let primes = primes_up_to(20);
        let expected = vec![2, 3, 5, 7, 11, 13, 17, 19];
        assert_eq!(primes, expected);
    }
}
