package api

import "fmt"

const maxTokensUpperBound = 4096

type CompletionsRequest struct {
	RequestId   string  `json:"request_id"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

func (r *CompletionsRequest) validate() string {
	if r.Prompt == "" {
		return "prompt must not be empty"
	}
	if r.MaxTokens <= 0 {
		return "max_tokens must be greater than 0"
	}
	if r.MaxTokens > maxTokensUpperBound {
		return fmt.Sprintf("max_tokens must not exceed %d", maxTokensUpperBound)
	}
	if r.Temperature < 0.0 || r.Temperature > 2.0 {
		return "temperature must be between 0.0 and 2.0"
	}
	return ""
}

type CompletionsReponse struct {
	RequestId       string `json:"request_id"`
	GeneratedText   string `json:"generated_text"`
	TokensGenerated int    `json:"tokens_generated"`
	InferenceTimeMs int    `json:"inference_time_ms"`
}

type ErrorResponse struct {
	RequestId string `json:"request_id,omitempty"`
	Error     string `json:"error"`
}
