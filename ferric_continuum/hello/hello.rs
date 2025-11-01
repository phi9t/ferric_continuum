use hello_lib_rs::{fibonacci, generate_greeting, primes_up_to};
use tracing::{info, Level};
use tracing_subscriber;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!(greeting = generate_greeting("World"), "Welcome");

    // Demonstrate fibonacci
    info!("Fibonacci sequence (first 10 numbers):");
    for i in 0..10 {
        info!(n = i, result = fibonacci(i), "fib");
    }

    // Demonstrate prime checking
    info!("Prime numbers up to 50:");
    let primes = primes_up_to(50);
    for prime in primes {
        info!(prime, "Found prime");
    }
}
