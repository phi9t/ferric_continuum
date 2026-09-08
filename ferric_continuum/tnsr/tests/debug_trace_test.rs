use tnsr::{
    autograd::Engine,
    ops::{basic, linear},
    tensor::{Shape, Tensor, TensorValue},
};

fn event_kinds(trace: &serde_json::Value) -> Vec<&str> {
    trace["events"]
        .as_array()
        .expect("events array")
        .iter()
        .map(|event| event["kind"].as_str().expect("event kind"))
        .collect()
}

#[test]
fn trace_json_exports_stable_forward_backward_schema() {
    let mut engine = Engine::new();
    let x = Tensor::from_value(
        TensorValue::from_vec(Shape(vec![2, 2]), vec![1.0, -2.0, 3.0, 4.0]),
        true,
    );
    let w = Tensor::from_value(
        TensorValue::from_vec(Shape(vec![2, 1]), vec![0.5, -1.5]),
        true,
    );

    let loss = engine.with_recording(|| {
        let y = linear::linear(&x, &w, "proj");
        let scaled = basic::scale(&y, 2.0, "double");
        basic::sum(&scaled, "loss")
    });

    engine.backward(&loss);

    let trace = engine.debug.trace_json();

    assert_eq!(trace["schema"], "tnsr.debug_trace");
    assert_eq!(trace["schema_version"], 1);

    let forward_ops = trace["forward_ops"].as_array().expect("forward_ops array");
    assert_eq!(forward_ops.len(), 3);

    let linear_op = forward_ops
        .iter()
        .find(|op| op["name"] == "proj")
        .expect("linear op");
    assert_eq!(linear_op["kind"], "Linear");
    assert_eq!(linear_op["inputs"].as_array().expect("inputs").len(), 2);
    assert_eq!(linear_op["outputs"].as_array().expect("outputs").len(), 1);
    assert_eq!(
        linear_op["edges"]["inputs"]
            .as_array()
            .expect("input edges")
            .len(),
        2
    );
    assert_eq!(
        linear_op["edges"]["outputs"]
            .as_array()
            .expect("output edges")
            .len(),
        1
    );
    assert!(linear_op["display_label"]
        .as_str()
        .expect("display label")
        .contains("Linear"));

    let saved_sites = linear_op["saved_sites"]
        .as_array()
        .expect("linear saved sites");
    assert_eq!(saved_sites.len(), 2);
    assert_eq!(saved_sites[0]["role"], "Activation");
    assert_eq!(saved_sites[0]["shape"], serde_json::json!([2, 2]));
    assert_eq!(saved_sites[0]["bytes"], 16);

    let kinds = event_kinds(&trace);
    assert!(kinds.contains(&"op"));
    assert!(kinds.contains(&"saved"));
    assert!(kinds.contains(&"saved_unpack"));
    assert!(kinds.contains(&"backward"));
    assert!(kinds.contains(&"grad_accum"));
    assert!(kinds.contains(&"grad_leaf_write"));

    assert!(trace["backward_ops"]
        .as_array()
        .expect("backward ops")
        .iter()
        .any(|event| event["op_kind"] == "Linear" && event["op_name"] == "proj"));
    assert!(trace["grad_accumulations"]
        .as_array()
        .expect("grad accumulations")
        .iter()
        .any(|event| event["kind"] == "grad_accum"
            && event["output"]["shape"] == serde_json::json!([2, 2])));

    let pretty = engine.debug.trace_json_pretty();
    assert!(pretty.contains("\"schema\": \"tnsr.debug_trace\""));
    assert!(pretty.contains("\"op_kind\": \"Linear\""));
}
