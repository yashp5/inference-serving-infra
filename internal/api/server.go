package api

import "net/http"

func NewMux(h *Handler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /infer", h.Infer)
	mux.HandleFunc("GET /health", h.Health)
	return mux
}
