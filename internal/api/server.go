package api

import "net/http"

// where do we buffer the incoming requests

func NewMux(h *Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /infer", h.Infer)
	mux.HandleFunc("GET /health", h.Health)
	return mux
}
