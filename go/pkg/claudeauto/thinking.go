package claudeauto

import "encoding/json"

// honorDisabledThinking makes an Anthropic `thinking: {"type": "disabled"}`
// effective. The bundled /v1/messages translation ignores that field, so a
// thinking-by-default template keeps reasoning and can spend the whole output
// budget before answering. The backend does honor
// chat_template_kwargs.enable_thinking=false, so map the one onto the other.
// An explicit enable_thinking from the client always wins, and bodies that do
// not disable thinking are returned byte-for-byte.
func honorDisabledThinking(body []byte) []byte {
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil {
		return body
	}
	var thinking struct {
		Type string `json:"type"`
	}
	if raw, ok := top["thinking"]; !ok || json.Unmarshal(raw, &thinking) != nil || thinking.Type != "disabled" {
		return body
	}
	kwargs := map[string]json.RawMessage{}
	if raw, ok := top["chat_template_kwargs"]; ok {
		if json.Unmarshal(raw, &kwargs) != nil {
			return body // Not an object: leave the client's value alone.
		}
		if _, set := kwargs["enable_thinking"]; set {
			return body
		}
	}
	kwargs["enable_thinking"] = json.RawMessage("false")
	encoded, err := json.Marshal(kwargs)
	if err != nil {
		return body
	}
	top["chat_template_kwargs"] = encoded
	out, err := json.Marshal(top)
	if err != nil {
		return body
	}
	return out
}
