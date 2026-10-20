//! Capture the real proxy HTTP boundary, with all upstream traffic confined to loopback.
use std::{sync::Arc, time::Duration};

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode, Uri},
    Router,
};
use codex_app_transfer_proxy::{
    build_router, ProviderResolver, ResolveError, ResolvedProvider, StaticResolver,
};
use codex_app_transfer_registry::{builtin_presets, Provider};
use serde_json::{json, Value};
use tokio::{net::TcpListener, sync::mpsc};

mod common;

struct LocalResolver {
    inner: StaticResolver,
    upstream: String,
}

impl ProviderResolver for LocalResolver {
    fn resolve(
        &self,
        parts: &axum::http::request::Parts,
        body: &[u8],
    ) -> Result<ResolvedProvider, ResolveError> {
        let mut resolved = self.inner.resolve(parts, body)?;
        // Keep the actual provider metadata and normal model-slot resolution; replace only
        // the network destination. No provider credential ever reaches this test.
        let suffix = resolved
            .provider
            .base_url
            .splitn(4, '/')
            .nth(3)
            .unwrap_or("");
        resolved.upstream_base = format!("{}/{}", self.upstream, suffix);
        Ok(resolved)
    }
}

struct Captured {
    uri: String,
    headers: HeaderMap,
    body: Value,
}

async fn capture(
    State(tx): State<mpsc::Sender<Captured>>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> StatusCode {
    tx.send(Captured {
        uri: uri.to_string(),
        headers,
        body: serde_json::from_slice(&body).unwrap(),
    })
    .await
    .unwrap();
    // Deliberate upstream error: inspect the outgoing request without inventing an inference response.
    StatusCode::BAD_REQUEST
}

fn preset(id: &str, host: &str) -> Provider {
    let value = builtin_presets()
        .iter()
        .find(|p| p["id"] == id)
        .unwrap()
        .clone();
    let mut p: Provider = serde_json::from_value(value).unwrap();
    p.base_url = p.base_url.replace("api.minimax.io", host);
    p.api_key = "offline-fixture-key".into();
    p
}

async fn send(p: Provider, input: Value, path: &str) -> (StatusCode, Option<Captured>) {
    let (tx, mut rx) = mpsc::channel(1);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let upstream = format!("http://{}", listener.local_addr().unwrap());
    let mock = tokio::spawn(async move {
        axum::serve(listener, Router::new().fallback(capture).with_state(tx))
            .await
            .unwrap();
    });
    let resolver = LocalResolver {
        inner: StaticResolver::new(None, vec![p.clone()], Some(p.id)),
        upstream,
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}{path}", listener.local_addr().unwrap());
    let proxy = tokio::spawn(async move {
        axum::serve(listener, build_router(Arc::new(resolver)))
            .await
            .unwrap();
    });
    let response = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap()
        .post(url)
        .header("content-type", "application/json")
        .body(input.to_string())
        .send()
        .await
        .unwrap();
    let status = response.status();
    let captured = rx.try_recv().ok();
    proxy.abort();
    mock.abort();
    (status, captured)
}

fn history() -> Value {
    json!({
        "model": "gpt-5.5",
        "stream": true,
        "input": [
            {"role":"user", "content":[
                {"type":"input_text", "text":"Inspect this image"},
                {"type":"input_image", "image_url":"https://example.invalid/image.png"}
            ]},
            {"type":"reasoning", "summary":[{"type":"summary_text", "text":"Inspect before replying."}]},
            {"type":"function_call", "call_id":"call_one", "name":"inspect", "arguments":"{\"path\":\"image.png\"}"},
            {"type":"function_call_output", "call_id":"call_one", "output":"image inspected"},
            {"role":"user", "content":"Continue"}
        ],
        "tools":[{"type":"function", "name":"inspect", "parameters":{"type":"object", "properties":{"path":{"type":"string"}}}}]
    })
}

#[tokio::test]
async fn minimax_flash_default_and_all_efforts_reach_http_with_image_and_tool_history() {
    for host in ["api.minimax.io", "api.minimaxi.com"] {
        for id in ["minimax", "minimax-anthropic"] {
            for effort in [
                None,
                Some("low"),
                Some("medium"),
                Some("high"),
                Some("xhigh"),
                Some("max"),
            ] {
                let mut input = history();
                if let Some(effort) = effort {
                    input["reasoning"] = json!({"effort": effort});
                }
                let (status, captured) = send(preset(id, host), input, "/v1/responses").await;
                assert_eq!(status, StatusCode::BAD_REQUEST); // mock's response
                let captured = captured.expect("request must reach mocked upstream");
                let body = captured.body;
                assert_eq!(body["model"], "MiniMax-M3.1-Flash", "{id} {host}");
                assert_eq!(body["thinking"], json!({"type":"adaptive"}));
                assert_eq!(
                    captured.headers["authorization"],
                    "Bearer offline-fixture-key"
                );
                assert!(body.get("previous_response_id").is_none());
                assert!(body.get("service_tier").is_none());
                assert_eq!(body["tools"].as_array().unwrap().len(), 1);
                if id == "minimax" {
                    assert_eq!(captured.uri, "/v1/chat/completions");
                    assert_eq!(body["reasoning_effort"], effort.unwrap_or("max"));
                    assert!(body.get("output_config").is_none());
                    let messages = body["messages"].as_array().unwrap();
                    assert!(messages.iter().any(|m| m["content"].as_array().is_some_and(
                        |blocks| blocks
                            .iter()
                            .any(|b| b["image_url"]["url"] == "https://example.invalid/image.png")
                    )));
                    assert!(messages
                        .iter()
                        .any(|m| m["reasoning_content"] == "Inspect before replying."));
                    assert!(messages
                        .iter()
                        .any(|m| m["tool_calls"][0]["id"] == "call_one"));
                    assert!(messages
                        .iter()
                        .any(|m| m["tool_call_id"] == "call_one"
                            && m["content"] == "image inspected"));
                } else {
                    assert_eq!(captured.uri, "/anthropic/v1/messages");
                    assert_eq!(captured.headers["anthropic-version"], "2023-06-01");
                    assert_eq!(body["output_config"]["effort"], effort.unwrap_or("max"));
                    assert!(body.get("reasoning_effort").is_none());
                    let blocks: Vec<&Value> = body["messages"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|m| m["content"].as_array().unwrap())
                        .collect();
                    assert!(blocks.iter().any(|b| b["type"] == "image"
                        && b["source"]["url"] == "https://example.invalid/image.png"));
                    assert!(blocks
                        .iter()
                        .any(|b| b["thinking"] == "Inspect before replying."));
                    assert!(blocks
                        .iter()
                        .any(|b| b["type"] == "tool_use" && b["id"] == "call_one"));
                    assert!(blocks
                        .iter()
                        .any(|b| b["type"] == "tool_result" && b["tool_use_id"] == "call_one"));
                }
            }
        }
    }
}

#[tokio::test]
async fn minimax_flash_rejects_unsupported_controls_before_network() {
    for id in ["minimax", "minimax-anthropic"] {
        for (key, value) in [
            ("thinking", json!({"type":"disabled"})),
            ("reasoning", json!({"effort":"none"})),
            ("reasoning", json!({"effort":"unknown"})),
            ("service_tier", json!("priority")),
            ("reasoning_split", json!(false)),
        ] {
            let mut input = history();
            input[key] = value;
            let (status, captured) =
                send(preset(id, "api.minimax.io"), input, "/v1/responses").await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert!(
                captured.is_none(),
                "unsupported {key} must not be sent to {id}"
            );
        }
    }
}

#[tokio::test]
async fn saved_legacy_custom_and_m3_model_mappings_are_not_reset() {
    for model in ["MiniMax-M2.7", "MiniMax-M3", "saved-custom-model"] {
        let mut p = preset("minimax", "api.minimax.io");
        p.models.insert("default".into(), model.into());
        let (_, captured) = send(
            p,
            json!({"model":"gpt-5.5", "input":"hello"}),
            "/v1/responses",
        )
        .await;
        assert_eq!(captured.unwrap().body["model"], model);
    }
}

#[tokio::test]
async fn m3_keeps_protocol_defaults_and_ignores_effort_depth() {
    for id in ["minimax", "minimax-anthropic"] {
        for effort in [None, Some("none"), Some("max")] {
            let mut p = preset(id, "api.minimax.io");
            p.models.insert("default".into(), "MiniMax-M3".into());
            let mut input = json!({"model":"gpt-5.5", "input":"hello"});
            if let Some(effort) = effort {
                input["reasoning"] = json!({"effort": effort});
            }
            let (_, captured) = send(p, input, "/v1/responses").await;
            let body = captured.unwrap().body;
            let enabled = effort.map_or(id == "minimax", |e| e != "none");
            assert_eq!(
                body["thinking"]["type"],
                if enabled { "adaptive" } else { "disabled" }
            );
            assert!(body.get("reasoning_effort").is_none());
            assert!(body.pointer("/output_config/effort").is_none());
        }
    }
}

#[tokio::test]
async fn direct_chat_keeps_full_history_and_custom_gateway_is_not_given_direct_controls() {
    let input = json!({"model":"MiniMax-M3.1-Flash", "reasoning_effort":"xhigh", "messages":[
        {"role":"assistant", "content":"", "reasoning_details":[{"type":"reasoning.text", "text":"opaque history"}],
        "tool_calls":[{"id":"call_one", "type":"function", "function":{"name":"inspect", "arguments":"{}"}}]},
        {"role":"tool", "tool_call_id":"call_one", "content":"done"}
    ]});
    let (_, captured) = send(
        preset("minimax", "api.minimax.io"),
        input.clone(),
        "/v1/chat/completions",
    )
    .await;
    let body = captured.unwrap().body;
    assert_eq!(body["messages"], input["messages"]);
    assert_eq!(body["reasoning_effort"], "xhigh");
    assert_eq!(body["thinking"]["type"], "adaptive");
    let mut p = preset("minimax", "api.minimax.io");
    p.base_url = "https://gateway.example/v1".into();
    let (_, captured) = send(p, input.clone(), "/v1/chat/completions").await;
    assert_eq!(captured.unwrap().body, input);
}
