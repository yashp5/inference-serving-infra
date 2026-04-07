package worker

import (
	inferencepb "github.com/yashp5/inference-serving-infra/gen"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func New(addr string) (inferencepb.InferenceClient, *grpc.ClientConn, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, nil, err
	}
	return inferencepb.NewInferenceClient(conn), conn, nil
}
