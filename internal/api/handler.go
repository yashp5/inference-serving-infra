package api

import (
	"container/heap"
	"context"
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

const (
	rateLimiterRequests = 10
	rateLimiterWindowMs = 100
	workerCount         = 3
)

type Priority int

const (
	LOW Priority = iota
	MEDIUM
	HIGH
)

type Response struct {
	body  *CompletionsReponse
	error error
}

type Request struct {
	body     *CompletionsRequest
	priority Priority
	respCh   chan *Response
	ctx      context.Context
}

type Handler struct {
	inferClient   inferencepb.InferenceClient
	conn          *grpc.ClientConn
	priorityQueue *PriorityQueue
	rateLimiter   RateLimiter
}

func NewHandler(ctx context.Context, inferClient inferencepb.InferenceClient, conn *grpc.ClientConn) *Handler {
	pq := NewPriorityQueue()
	wchs := make([]chan *Request, 0)
	for range workerCount {
		worker := NewWorker(inferClient)
		worker.Start(ctx)
		wchs = append(wchs, worker.reqch)
	}
	dispatcher := NewDispatcher(pq, wchs)
	dispatcher.Start(ctx)
	return &Handler{
		inferClient:   inferClient,
		conn:          conn,
		rateLimiter:   NewTokenBucketRateLimiter(rateLimiterRequests, rateLimiterWindowMs),
		priorityQueue: pq,
	}
}

func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	state := h.conn.GetState()
	if state == connectivity.Shutdown || state == connectivity.TransientFailure {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "worker_unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func (h *Handler) Infer(w http.ResponseWriter, r *http.Request) {
	if !h.rateLimiter.allow(r.RemoteAddr) {
		writeJSON(w, http.StatusTooManyRequests, "rate limited")
		return
	}
	id, _ := uuid.NewV7()
	requestId := id.String()

	reqBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{RequestId: requestId, Error: "failed to read request body"})
		return
	}
	defer r.Body.Close()

	reqBody := &CompletionsRequest{}
	if err := json.Unmarshal(reqBytes, reqBody); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{RequestId: requestId, Error: "failed to unmarshal req body"})
		return
	}

	if errMsg := reqBody.validate(); errMsg != "" {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{RequestId: requestId, Error: errMsg})
		return
	}
	req := &Request{
		body:     reqBody,
		priority: MEDIUM,
		ctx:      r.Context(),
		respCh:   make(chan *Response),
	}
	h.priorityQueue.Push(req)

	select {
	case <-time.After(5 * time.Second):
		writeJSON(w, http.StatusRequestTimeout, ErrorResponse{RequestId: requestId, Error: "request timeout"})
	case resp := <-req.respCh:
		if resp.error != nil {
			writeJSON(w, http.StatusRequestTimeout, ErrorResponse{RequestId: requestId, Error: resp.error.Error()})
			return
		}
		writeJSON(w, http.StatusOK, resp.body)
	}
}

type Dispatcher struct {
	pq        *PriorityQueue
	workerChs []chan *Request
	next      int
}

func NewDispatcher(pq *PriorityQueue, workerChs []chan *Request) *Dispatcher {
	return &Dispatcher{
		pq:        pq,
		workerChs: workerChs,
		next:      0,
	}
}

func (d *Dispatcher) Start(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-d.pq.signal:
				d.pq.mu.Lock()
				if d.pq.buf.Len() == 0 {
					d.pq.mu.Unlock()
					continue
				}
				reqs := make([]*Request, 0)
				for d.pq.buf.Len() > 0 {
					req := heap.Pop(&d.pq.buf).(*Request)
					reqs = append(reqs, req)
				}
				d.pq.mu.Unlock()
				for _, req := range reqs {
					d.workerChs[d.next%len(d.workerChs)] <- req
					d.next++
				}
			}
		}
	}()
}

const workerBufferSize = 100

type Worker struct {
	inferClient inferencepb.InferenceClient
	reqch       chan *Request
}

func NewWorker(inferClient inferencepb.InferenceClient) *Worker {
	return &Worker{
		inferClient: inferClient,
		reqch:       make(chan *Request, workerBufferSize),
	}
}

func (w *Worker) Start(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case req := <-w.reqch:
				w.Process(*req)
			}
		}
	}()
}

func (w *Worker) Process(req Request) {
	totalStart := time.Now()

	grpcStart := time.Now()
	inferResp, err := w.inferClient.Generate(req.ctx, &inferencepb.GenerateRequest{
		RequestId:   req.body.RequestId,
		Prompt:      req.body.Prompt,
		MaxTokens:   int32(req.body.MaxTokens),
		Temperature: float32(req.body.Temperature),
	})
	grpcLatency := time.Since(grpcStart)
	totalLatency := time.Since(totalStart)

	if err != nil {
		log.Printf("[request_id=%s] grpc error total=%dms grpc=%dms overhead=%dms err=%v", req.body.RequestId, totalLatency.Milliseconds(), grpcLatency.Milliseconds(), (totalLatency - grpcLatency).Milliseconds(), err)
		req.respCh <- &Response{
			error: err,
		}
		return
	}

	respBody := &CompletionsReponse{
		RequestId:       req.body.RequestId,
		GeneratedText:   inferResp.GeneratedText,
		TokensGenerated: int(inferResp.TokensGenerated),
		InferenceTimeMs: int(inferResp.InferenceTimeMs),
	}
	req.respCh <- &Response{
		body: respBody,
	}
}
