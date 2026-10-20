//! Direct MiniMax text wire controls. Gateway contracts are intentionally separate.
//! Source: https://platform.minimax.io/docs/api-reference/text-openai-api
//! Source: https://platform.minimax.io/docs/api-reference/text-anthropic-api
use codex_app_transfer_registry::Provider;
use serde_json::{json, Map, Value};

use crate::AdapterError;

pub(crate) fn is_direct(provider: &Provider) -> bool {
    matches!(
        provider.base_url.split('/').nth(2),
        Some("api.minimax.io" | "api.minimaxi.com")
    )
}

pub(crate) fn apply_controls(
    target: &mut Map<String, Value>,
    source: &Value,
    provider: &Provider,
    anthropic: bool,
) -> Result<(), AdapterError> {
    if !is_direct(provider) {
        return Ok(());
    }
    match target.get("model").and_then(Value::as_str) {
        Some("MiniMax-M3") => {
            // M3 has an on/off switch, not Flash's depth control. Defaults differ by API.
            let thinking = match source.get("thinking") {
                Some(value)
                    if matches!(
                        value.get("type").and_then(Value::as_str),
                        Some("adaptive" | "disabled")
                    ) =>
                {
                    value.clone()
                }
                Some(_) => {
                    return Err(AdapterError::BadRequest(
                        "MiniMax-M3 thinking must be adaptive or disabled".into(),
                    ))
                }
                None => {
                    let effort = source
                        .pointer("/reasoning/effort")
                        .or_else(|| source.get("reasoning_effort"))
                        .or_else(|| source.pointer("/output_config/effort"))
                        .and_then(Value::as_str);
                    let enabled =
                        effort.map_or(!anthropic, |e| !matches!(e, "none" | "off" | "disabled"));
                    json!({"type": if enabled { "adaptive" } else { "disabled" }})
                }
            };
            target.insert("thinking".into(), thinking);
            target.remove("reasoning_effort");
            if let Some(output) = target
                .get_mut("output_config")
                .and_then(Value::as_object_mut)
            {
                output.remove("effort");
            }
            return Ok(());
        }
        Some("MiniMax-M3.1-Flash") => {}
        _ => return Ok(()),
    }
    if source.get("service_tier").is_some()
        || source.get("reasoning_split") == Some(&Value::Bool(false))
    {
        return Err(AdapterError::BadRequest(
            "MiniMax-M3.1-Flash does not support service_tier or reasoning_split=false".into(),
        ));
    }
    if let Some(thinking) = source.get("thinking") {
        if thinking.get("type").and_then(Value::as_str) != Some("adaptive") {
            return Err(AdapterError::BadRequest(
                "MiniMax-M3.1-Flash requires adaptive thinking".into(),
            ));
        }
    }
    let requested = if anthropic {
        source.pointer("/output_config/effort")
    } else {
        source.get("reasoning_effort")
    }
    .or_else(|| source.pointer("/reasoning/effort"));
    let effort = match requested {
        None => "max",
        Some(Value::String(s))
            if matches!(s.as_str(), "low" | "medium" | "high" | "xhigh" | "max") =>
        {
            s
        }
        _ => {
            return Err(AdapterError::BadRequest(
                "MiniMax-M3.1-Flash effort must be low, medium, high, xhigh, or max".into(),
            ))
        }
    };
    target.insert("thinking".into(), json!({"type": "adaptive"}));
    target.remove("reasoning");
    if anthropic {
        target.remove("reasoning_effort");
        let output = target.entry("output_config").or_insert_with(|| json!({}));
        let output = output
            .as_object_mut()
            .ok_or_else(|| AdapterError::BadRequest("output_config must be an object".into()))?;
        output.insert("effort".into(), json!(effort));
    } else {
        target.remove("output_config");
        target.insert("reasoning_effort".into(), json!(effort));
    }
    Ok(())
}
