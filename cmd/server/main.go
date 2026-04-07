package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/yashp5/inference-serving-infra/internal/api"
	"github.com/yashp5/inference-serving-infra/internal/worker"
)

func main() {
	inferClient, conn, err := worker.New("localhost:50051")
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	h := api.NewHandler(inferClient, conn)
	mux := api.NewMux(h)

	srv := &http.Server{Addr: ":8080", Handler: mux}
	go srv.ListenAndServe()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	srv.Shutdown(ctx)
}
