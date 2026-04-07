package api

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"
	inferencepb "github.com/yashp5/inference-serving-infra/gen"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
)

type Handler struct {
	inferClient inferencepb.InferenceClient
	conn        *grpc.ClientConn
}

func NewHandler(inferClient inferencepb.InferenceClient, conn *grpc.ClientConn) *Handler {
	return &Handler{
		inferClient: inferClient,
		conn:        conn,
	}
}

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	state := h.conn.GetState()
	if state == connectivity.Shutdown || state == connectivity.TransientFailure {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "worker_unavailable"})
		return
	}
	writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (h *Handler) Infer(w http.ResponseWriter, r *http.Request) {
	id, _ := uuid.NewV7()
	requestId := id.String()

	totalStart := time.Now()

	reqBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{RequestId: requestId, Error: "failed to read request body"})
		return
	}
	defer r.Body.Close()

	req := &CompletionsRequest{}
	if err := json.Unmarshal(reqBytes, req); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{RequestId: requestId, Error: "failed to unmarshal req body"})
		return
	}

	if errMsg := req.validate(); errMsg != "" {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "ok"})
	}

	grpcStart := time.Now()
	inferResp, err := h.inferClient.Generate(r.Context(), &inferencepb.GenerateRequest{
		RequestId:   requestId,
		Prompt:      req.Prompt,
		MaxTokens:   int32(req.MaxTokens),
		Temperature: float32(req.Temperature),
	})
	grpcLatency := time.Since(grpcStart)
	totalLatency := time.Since(totalStart)

	if err != nil {
		log.Printf("[request_id=%s] grpc error total=%dms grpc=%dms overhead=%dms err=%v", requestId, totalLatency.Milliseconds(), grpcLatency.Milliseconds(), (totalLatency - grpcLatency).Milliseconds(), err)
		writeJSON(w, http.StatusBadGateway, ErrorResponse{RequestId: requestId, Error: "inference failed: " + err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, CompletionsReponse{
		RequestId:       requestId,
		GeneratedText:   inferResp.GeneratedText,
		TokensGenerated: int(inferResp.InferenceTimeMs),
		InferenceTimeMs: int(inferResp.InferenceTimeMs),
	})
}
